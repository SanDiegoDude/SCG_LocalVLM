import os
import gc
import inspect
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
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
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
                    ],
                    {"default": "Qwen3-VL-4B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        text,
        model,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
        image=None,
        video_path=None,
    ):
        if seed != -1:
            torch.manual_seed(seed)

        if model.startswith("Qwen"):
            model_id = f"qwen/{model}"
        else:
            model_id = f"Skywork/{model}"
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
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            # Choose the appropriate model class based on the model family
            if model.startswith("Qwen3"):
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_checkpoint,
                    torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                    device_map="auto",
                    quantization_config=quantization_config,
                )
            else:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_checkpoint,
                    torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                    device_map="auto",
                    quantization_config=quantization_config,
                )

        processed_video_path = None
        result = None

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]

            try:
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

                    messages[0]["content"].insert(0, {
                        "type": "video",
                        "video": processed_video_path,
                    })
                else:
                    print("deal image")
                    pil_image = tensor_to_pil(image)
                    messages[0]["content"].insert(0, {
                        "type": "image",
                        "image": pil_image,
                    })

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
                ).to("cuda")

                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    temperature=temperature,
                )
            except Exception as e:
                return (f"Error during model inference: {str(e)}",)
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
                    [
                        "Qwen2.5-3B-Instruct",
                        "Qwen2.5-7B-Instruct",
                        "Qwen2.5-14B-Instruct",
                        "Qwen2.5-32B-Instruct",
                        "Qwen3-4B-Thinking-2507",
                        "Qwen3-4B-Instruct-2507"
                    ],
                    {"default": "Qwen3-4B-Instruct-2507"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),  # add quantization type selection
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1}),  # add seed parameter, default is -1
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "inference"
    CATEGORY = "Comfyui_QwenVL"

    def inference(
        self,
        system,
        prompt,
        model,
        quantization,
        keep_model_loaded,
        temperature,
        max_new_tokens,
        seed,
    ):
        if not prompt.strip() and not system.strip():
            return ("Error: Both system and prompt are empty.",)

        if seed != -1:
            torch.manual_seed(seed)
        model_id = f"qwen/{model}"
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
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )

        result = None
        with torch.no_grad():
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ]

            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                inputs = self.tokenizer([text], return_tensors="pt").to("cuda")

                generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                result = self.tokenizer.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    temperature=temperature,
                )
            except Exception as e:
                return (f"Error during model inference: {str(e)}",)
            finally:
                if not keep_model_loaded:
                    self._unload_resources()

            return result
