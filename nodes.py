import os
import gc
import inspect
import json
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info
from PIL import Image
import numpy as np
import folder_paths
import subprocess
import uuid

try:
    import comfy.model_management as comfy_mm
except ImportError:  # ComfyUI runtime not available during development/tests
    comfy_mm = None


# --- Custom Models Loading ---
CUSTOM_MODELS_FILE = os.path.join(os.path.dirname(__file__), "custom_models.json")
CUSTOM_VL_MODELS = {}  # name -> {repo_id, model_class, ...}
CUSTOM_TEXT_MODELS = {}  # name -> {repo_id, ...}


def _load_custom_models():
    """Load custom models from custom_models.json if it exists."""
    global CUSTOM_VL_MODELS, CUSTOM_TEXT_MODELS
    if not os.path.exists(CUSTOM_MODELS_FILE):
        return
    
    try:
        with open(CUSTOM_MODELS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        hf_models = data.get("hf_models", {})
        for model_name, model_info in hf_models.items():
            repo_id = model_info.get("repo_id", "")
            if not repo_id:
                continue
            
            # Determine if VL (vision-language) or text-only based on name
            is_vl = "-VL-" in model_name.upper() or "-VL-" in repo_id.upper()
            
            # Determine model class from name pattern
            model_class = model_info.get("model_class", None)
            if model_class is None:
                if "Qwen3" in model_name or "Qwen3" in repo_id:
                    model_class = "Qwen3"
                else:
                    model_class = "Qwen2.5"
            
            model_entry = {
                "repo_id": repo_id,
                "model_class": model_class,
                "default": model_info.get("default", False),
                "quantized": model_info.get("quantized", False),
                "vram_requirement": model_info.get("vram_requirement", {}),
            }
            
            if is_vl:
                CUSTOM_VL_MODELS[model_name] = model_entry
            else:
                CUSTOM_TEXT_MODELS[model_name] = model_entry
        
        if CUSTOM_VL_MODELS or CUSTOM_TEXT_MODELS:
            print(f"[SCG_LocalVLM] Loaded {len(CUSTOM_VL_MODELS)} custom VL models and {len(CUSTOM_TEXT_MODELS)} custom text models from custom_models.json")
    except Exception as e:
        print(f"[SCG_LocalVLM] Warning: Failed to load custom_models.json: {e}")


# Load custom models at startup
_load_custom_models()


def _get_model_repo_id(model_name, is_vl=True):
    """Get the HuggingFace repo ID for a model name."""
    # Check custom models first
    if is_vl and model_name in CUSTOM_VL_MODELS:
        return CUSTOM_VL_MODELS[model_name]["repo_id"]
    if not is_vl and model_name in CUSTOM_TEXT_MODELS:
        return CUSTOM_TEXT_MODELS[model_name]["repo_id"]
    
    # Built-in models
    if model_name.startswith("Qwen"):
        return f"qwen/{model_name}"
    elif model_name == "SkyCaptioner-V1":
        return f"Skywork/{model_name}"
    else:
        return f"qwen/{model_name}"


def _get_model_class(model_name, is_vl=True):
    """Determine which model class to use for loading."""
    # Check custom models first
    if is_vl and model_name in CUSTOM_VL_MODELS:
        return CUSTOM_VL_MODELS[model_name].get("model_class", "Qwen3")
    if not is_vl and model_name in CUSTOM_TEXT_MODELS:
        return CUSTOM_TEXT_MODELS[model_name].get("model_class", "Qwen3")
    
    # Built-in: detect from name
    if model_name.startswith("Qwen3"):
        return "Qwen3"
    return "Qwen2.5"


# Build model lists including custom models
BUILTIN_VL_MODELS = [
    "Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct",
    "Qwen3-VL-2B-Thinking",
    "Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Thinking",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-32B-Thinking",
    "Qwen3-VL-32B-Instruct",
    "SkyCaptioner-V1",
]

BUILTIN_TEXT_MODELS = [
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-14B-Instruct",
    "Qwen2.5-32B-Instruct",
    "Qwen3-4B-Thinking-2507",
    "Qwen3-4B-Instruct-2507",
]


def _get_vl_model_list():
    """Get the full list of VL models including custom ones."""
    return BUILTIN_VL_MODELS + list(CUSTOM_VL_MODELS.keys())


def _get_text_model_list():
    """Get the full list of text models including custom ones."""
    return BUILTIN_TEXT_MODELS + list(CUSTOM_TEXT_MODELS.keys())


def _maybe_move_to_cpu(module):
    if module is None:
        return
    try:
        module.to("cpu")
    except Exception:
        pass


def _clear_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if comfy_mm is not None:
        try:
            soft_empty = getattr(comfy_mm, "soft_empty_cache", None)
            if callable(soft_empty):
                params = inspect.signature(soft_empty).parameters
                if "force" in params:
                    soft_empty(force=True)
                else:
                    soft_empty()
                return
            cleanup_models = getattr(comfy_mm, "cleanup_models", None)
            if callable(cleanup_models):
                cleanup_models()
        except Exception:
            pass


def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    image_tensor = image_tensor[batch_index].unsqueeze(0)
    i = 255.0 * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8).squeeze())
    return img


class QwenVL:
    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    def _unload_resources(self):
        _maybe_move_to_cpu(self.model)
        self.model = None
        self.processor = None
        _clear_cuda_memory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                    },
                ),
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    _get_vl_model_list(),
                    {"default": "Qwen3-VL-4B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "bypass": ("BOOLEAN", {"default": False}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 2, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "top_k": (
                    "INT",
                    {"default": 50, "min": 0, "max": 200, "step": 1},
                ),
                "min_p": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 8000, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        system,
        text,
        model,
        quantization,
        keep_model_loaded,
        bypass,
        do_sample,
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        max_new_tokens,
        seed,
        image1=None,
        image2=None,
        image3=None,
        image4=None,
        video_path=None,
    ):
        # Bypass mode: pass text directly to output without model inference
        if bypass:
            return (text,)

        if seed != -1:
            torch.manual_seed(seed)

        model_id = _get_model_repo_id(model, is_vl=True)
        # put downloaded model to model/LLM dir
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
            )

        if self.processor is None:
            # Define min_pixels and max_pixels:
            # Images will be resized to maintain their aspect ratio
            # within the range of min_pixels and max_pixels.
            min_pixels = 256*28*28
            max_pixels = 1024*28*28 

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if self.model is None:
            # Determine compute dtype for quantization
            compute_dtype = torch.bfloat16 if self.bf16_support else torch.float16

            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": "auto",
                }
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=compute_dtype,
                )
                load_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": "auto",
                }
            else:
                # No quantization - load with optimal settings
                load_kwargs = {
                    "torch_dtype": compute_dtype,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                }

            # Try to use Flash Attention 2 for better performance (optional)
            if torch.cuda.is_available():
                try:
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    # Flash Attention not available, will use default
                    pass

            # Choose the appropriate model class based on the model family
            model_class = _get_model_class(model, is_vl=True)
            if model_class == "Qwen3":
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_checkpoint,
                    **load_kwargs,
                )
            else:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_checkpoint,
                    **load_kwargs,
                )

        processed_video_path = None
        result = None

        with torch.inference_mode():
            # Build user content list - images first (in order), then text
            user_content = []
            
            # Process images in order: image1, image2, image3, image4
            images = [image1, image2, image3, image4]
            image_count = 0
            for idx, img in enumerate(images, start=1):
                if img is not None:
                    print(f"Processing image{idx}")
                    pil_image = tensor_to_pil(img)
                    user_content.append({
                        "type": "image",
                        "image": pil_image,
                    })
                    image_count += 1
            
            if image_count > 0:
                print(f"Total images added: {image_count}")

            try:
                # Handle video if provided (takes precedence positioning but images still included)
                if video_path:
                    print("deal video_path", video_path)
                    unique_id = uuid.uuid4().hex  # 生成唯一标识符
                    processed_video_path = f"/tmp/processed_video_{unique_id}.mp4"  # 临时文件路径
                    ffmpeg_command = [
                        "ffmpeg",
                        "-i", video_path,
                        "-vf", "fps=1,scale='min(256,iw)':min'(256,ih)':force_original_aspect_ratio=decrease",
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "18",
                        processed_video_path
                    ]
                    subprocess.run(ffmpeg_command, check=True)

                    user_content.append({
                        "type": "video",
                        "video": processed_video_path,
                    })
                
                # Add text prompt at the end
                user_content.append({"type": "text", "text": text})
            
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                print("deal messages", messages)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)

                # Build generation kwargs with proper parameter handling
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                }

                # Add sampling parameters only if do_sample is True
                if do_sample:
                    if temperature > 0:
                        generation_kwargs["temperature"] = temperature
                    if top_p < 1.0:
                        generation_kwargs["top_p"] = top_p
                    if top_k > 0:
                        generation_kwargs["top_k"] = top_k
                    if min_p > 0:
                        generation_kwargs["min_p"] = min_p

                # Always apply repetition penalty if not default
                if repetition_penalty != 1.0:
                    generation_kwargs["repetition_penalty"] = repetition_penalty

                generated_ids = self.model.generate(**inputs, **generation_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except torch.cuda.OutOfMemoryError as e:
                _clear_cuda_memory()
                return (
                    f"[SCG_LocalVLM] CUDA Out of Memory Error. "
                    f"Try: 1) Enable quantization (4bit/8bit), "
                    f"2) Reduce max_new_tokens, "
                    f"3) Use a smaller model. "
                    f"Error: {str(e)}",
                )
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"[SCG_LocalVLM] Error during inference:\n{error_details}")
                return (f"[SCG_LocalVLM] Error during model inference: {str(e)}",)
            finally:
                if not keep_model_loaded:
                    self._unload_resources()
                if processed_video_path:
                    try:
                        os.remove(processed_video_path)
                    except FileNotFoundError:
                        pass

            return result


class Qwen:
    def __init__(self):
        self.model_checkpoint = None
        self.tokenizer = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    def _unload_resources(self):
        _maybe_move_to_cpu(self.model)
        self.model = None
        self.tokenizer = None
        _clear_cuda_memory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "system": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant.",
                        "multiline": True,
                    },
                ),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    _get_text_model_list(),
                    {"default": "Qwen3-4B-Instruct-2507"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "bypass": ("BOOLEAN", {"default": False}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 2, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "top_k": (
                    "INT",
                    {"default": 50, "min": 0, "max": 200, "step": 1},
                ),
                "min_p": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 8000, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        system,
        prompt,
        model,
        quantization,
        keep_model_loaded,
        bypass,
        do_sample,
        temperature,
        top_p,
        top_k,
        min_p,
        repetition_penalty,
        max_new_tokens,
        seed,
    ):
        # Bypass mode: pass prompt directly to output without model inference
        if bypass:
            return (prompt,)

        if not prompt.strip() and not system.strip():
            return ("Error: Both system and prompt are empty.",)

        if seed != -1:
            torch.manual_seed(seed)
        model_id = _get_model_repo_id(model, is_vl=False)
        # put downloaded model to model/LLM dir
        self.model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", os.path.basename(model_id)
        )

        if not os.path.exists(self.model_checkpoint):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=model_id,
                local_dir=self.model_checkpoint,
            )

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        if self.model is None:
            # Determine compute dtype for quantization
            compute_dtype = torch.bfloat16 if self.bf16_support else torch.float16

            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": "auto",
                }
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=compute_dtype,
                )
                load_kwargs = {
                    "quantization_config": quantization_config,
                    "device_map": "auto",
                }
            else:
                # No quantization - load with optimal settings
                load_kwargs = {
                    "torch_dtype": compute_dtype,
                    "device_map": "auto",
                    "low_cpu_mem_usage": True,
                }

            # Try to use Flash Attention 2 for better performance (optional)
            if torch.cuda.is_available():
                try:
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                except Exception:
                    # Flash Attention not available, will use default
                    pass

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                **load_kwargs,
            )

        result = None
        with torch.inference_mode():
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

                # Build generation kwargs with proper parameter handling
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                }

                # Add sampling parameters only if do_sample is True
                if do_sample:
                    if temperature > 0:
                        generation_kwargs["temperature"] = temperature
                    if top_p < 1.0:
                        generation_kwargs["top_p"] = top_p
                    if top_k > 0:
                        generation_kwargs["top_k"] = top_k
                    if min_p > 0:
                        generation_kwargs["min_p"] = min_p

                # Always apply repetition penalty if not default
                if repetition_penalty != 1.0:
                    generation_kwargs["repetition_penalty"] = repetition_penalty

                generated_ids = self.model.generate(**inputs, **generation_kwargs)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            except torch.cuda.OutOfMemoryError as e:
                _clear_cuda_memory()
                return (
                    f"[SCG_LocalVLM] CUDA Out of Memory Error. "
                    f"Try: 1) Enable quantization (4bit/8bit), "
                    f"2) Reduce max_new_tokens, "
                    f"3) Use a smaller model. "
                    f"Error: {str(e)}",
                )
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"[SCG_LocalVLM] Error during inference:\n{error_details}")
                return (f"[SCG_LocalVLM] Error during model inference: {str(e)}",)
            finally:
                if not keep_model_loaded:
                    self._unload_resources()

            return result
