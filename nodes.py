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
import time  # Import at module level for timing

try:
    import comfy.model_management as comfy_mm
except ImportError:  # ComfyUI runtime not available during development/tests
    comfy_mm = None
def _check_quantization_support():
    """Check if quantization kernels are available and working."""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    try:
        import bitsandbytes as bnb
        # Test if CUDA kernels are available
        test_tensor = torch.randn(10, 10).cuda()
        try:
            # Try to use quantization kernel
            from bitsandbytes.functional import quantize_4bit
            quantize_4bit(test_tensor)
            return True, "Quantization kernels available"
        except Exception as e:
            return False, f"Quantization kernels not working: {e}"
    except ImportError:
        return False, "bitsandbytes not installed"


def _get_gpu_info():
    """Get GPU compute capability and check Flash Attention 2 optimization status.
    
    Returns info about whether FA2 is likely OPTIMAL for this GPU (hint, not blocker).
    FA2 may still work on newer GPUs even if not fully optimized yet.
    """
    if not torch.cuda.is_available():
        return None, None, True  # Assume compatible if we can't check
    
    try:
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        major, minor = props.major, props.minor
        compute_cap = f"{major}.{minor}"
        gpu_name = props.name
        
        # FA2 is optimized for SM 80-90 (Ampere, Ada, Hopper)
        # Newer architectures (SM 120+ Blackwell) may not have optimized kernels yet
        # But this could change with future FA2 updates
        sm_version = major * 10 + minor
        fa2_optimized = 80 <= sm_version <= 90
        
        return gpu_name, compute_cap, fa2_optimized
    except Exception:
        return None, None, True  # Assume compatible if we can't check


# Get GPU info at module load time
_GPU_NAME, _GPU_COMPUTE_CAP, _FA2_OPTIMIZED = _get_gpu_info()
if _GPU_NAME:
    print(f"[SCG_LocalVLM] GPU: {_GPU_NAME} (SM {_GPU_COMPUTE_CAP})")



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
    """Convert ComfyUI tensor to PIL Image with minimal CPU transfer."""
    # Extract single image from batch
    image_tensor = image_tensor[batch_index]

    # Perform scaling and clamping on GPU before CPU transfer
    img_data = (image_tensor * 255.0).clamp(0, 255).byte()

    # Single CPU transfer at the very end
    img_np = img_data.cpu().numpy()

    # Convert to PIL
    img = Image.fromarray(img_np)
    return img


class QwenVL:
    """
    Qwen Vision-Language Model Node for ComfyUI

    Supports Qwen2.5-VL and Qwen3-VL models with:
    - Multi-image input (up to 4 images)
    - Video input via FFmpeg processing
    - Multiple quantization options (none, 4bit, 8bit)
    - Advanced generation parameters
    - Flash Attention 2 optimization

    ComfyUI v2/v3 compatible.
    """

    def __init__(self):
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.current_image_resolution = None  # Track current resolution setting
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
        self.current_image_resolution = None
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
                "attention_mode": (
                    ["auto", "flash_attention_2", "sdpa", "eager"],
                    {"default": "auto"},
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
                "image_resolution": (
                    ["low (256x28x28)", "medium (512x28x28)", "high (768x28x28)", "ultra (1024x28x28)"],
                    {"default": "medium (512x28x28)"},
                ),
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
        attention_mode,
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
        image_resolution,
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

        # Recreate processor if it doesn't exist OR if resolution setting changed
        if self.processor is None or self.current_image_resolution != image_resolution:
            # Define min_pixels and max_pixels based on image_resolution setting:
            # Images will be resized to maintain their aspect ratio
            # within the range of min_pixels and max_pixels.
            # Lower resolution = faster processing, higher resolution = better quality
            resolution_map = {
                "low (256x28x28)": (128*28*28, 256*28*28),      # 100K - 200K pixels (fastest)
                "medium (512x28x28)": (256*28*28, 512*28*28),    # 200K - 400K pixels (balanced)
                "high (768x28x28)": (512*28*28, 768*28*28),      # 400K - 600K pixels (quality)
                "ultra (1024x28x28)": (768*28*28, 1024*28*28),   # 600K - 800K pixels (original, slowest)
            }
            min_pixels, max_pixels = resolution_map.get(image_resolution, (256*28*28, 512*28*28))

            if self.current_image_resolution != image_resolution:
                print(f"[SCG_LocalVLM] Image resolution changed: {self.current_image_resolution} → {image_resolution}")
            print(f"[SCG_LocalVLM] Image resolution: {image_resolution}")
            print(f"[SCG_LocalVLM] min_pixels={min_pixels:,}, max_pixels={max_pixels:,}")

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            self.current_image_resolution = image_resolution

        if self.model is None:
            load_start = time.time()
            
            # Check quantization support if using quantization
            if quantization != "none":
                quant_ok, quant_msg = _check_quantization_support()
                if not quant_ok:
                    print(f"[SCG_LocalVLM] WARNING: {quant_msg}")
                    print("[SCG_LocalVLM] Falling back to unquantized loading")
                    quantization = "none"
                else:
                    print(f"[SCG_LocalVLM] {quant_msg}")
            
            # Determine compute dtype
            compute_dtype = torch.bfloat16 if self.bf16_support else torch.float16
            print(f"[SCG_LocalVLM] Using compute dtype: {compute_dtype}")

            # Build load_kwargs - SIMPLIFIED for maximum performance
            # For non-quantized: use native PyTorch loading (no device_map overhead)
            # For quantized: need device_map for bitsandbytes
            load_kwargs = {
                "torch_dtype": compute_dtype,
            }
            
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = quantization_config
                # Use explicit GPU placement to avoid Accelerate memory management overhead
                load_kwargs["device_map"] = {"": 0} if torch.cuda.device_count() == 1 else "auto"
                print(f"[SCG_LocalVLM] Loading with 4-bit quantization")
                
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=compute_dtype,
                )
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = {"": 0} if torch.cuda.device_count() == 1 else "auto"
                print(f"[SCG_LocalVLM] Loading with 8-bit quantization")
                
            else:
                # Non-quantized: load directly to GPU for maximum speed
                load_kwargs["device_map"] = {"": "cuda:0"}
                print("[SCG_LocalVLM] Loading without quantization")

            # Handle attention mode
            if attention_mode == "flash_attention_2":
                try:
                    import flash_attn
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    attention_used = "flash_attention_2"
                    if not _FA2_OPTIMIZED:
                        print(f"[SCG_LocalVLM] Using Flash Attention 2 (may not be optimal for {_GPU_NAME or 'this GPU'})")
                    else:
                        print("[SCG_LocalVLM] Using Flash Attention 2")
                except ImportError:
                    print("[SCG_LocalVLM] flash-attn not installed, using SDPA instead")
                    load_kwargs["attn_implementation"] = "sdpa"
                    attention_used = "sdpa"
            elif attention_mode == "sdpa":
                load_kwargs["attn_implementation"] = "sdpa"
                attention_used = "sdpa"
                print("[SCG_LocalVLM] Using SDPA attention")
            elif attention_mode == "eager":
                load_kwargs["attn_implementation"] = "eager"
                attention_used = "eager"
                print("[SCG_LocalVLM] Using eager attention")
            else:
                # auto - use sdpa as default (best compatibility + good performance)
                load_kwargs["attn_implementation"] = "sdpa"
                attention_used = "sdpa"
                print("[SCG_LocalVLM] Using SDPA attention (auto)")

            # Load the model
            model_class = _get_model_class(model, is_vl=True)
            print(f"[SCG_LocalVLM] Loading {model_class} model from {self.model_checkpoint}")
            
            try:
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
                
                load_time = time.time() - load_start
                
                # Set model to eval mode
                self.model.eval()

                # CRITICAL: Clear CUDA cache and reset internal state to avoid random slowness
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    # Reset any accumulated gradients or state
                    for param in self.model.parameters():
                        param.grad = None

                # Print comprehensive model info
                print(f"[SCG_LocalVLM] Model loaded successfully in {load_time:.2f}s")
                print(f"[SCG_LocalVLM]   Device: {self.model.device}")
                print(f"[SCG_LocalVLM]   Dtype: {self.model.dtype}")
                print(f"[SCG_LocalVLM]   Attention: {attention_used}")

                # Verify actual attention implementation
                if hasattr(self.model.config, '_attn_implementation'):
                    actual_attn = self.model.config._attn_implementation
                    print(f"[SCG_LocalVLM]   Attention (verified): {actual_attn}")
                    if actual_attn != attention_used:
                        print(f"[SCG_LocalVLM]   WARNING: Requested {attention_used} but got {actual_attn}!")

                # GPU memory diagnostics
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                    print(f"[SCG_LocalVLM]   GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

                # Verify model is on GPU for non-quantized
                if quantization == "none":
                    # Check first parameter device
                    first_param = next(self.model.parameters())
                    if not first_param.is_cuda:
                        print(f"[SCG_LocalVLM]   WARNING: Model is on {first_param.device}, not CUDA!")
                        print(f"[SCG_LocalVLM]   This will cause VERY slow inference!")

                # Show quantization info
                if quantization != "none":
                    print(f"[SCG_LocalVLM]   Quantization: {quantization}")
                    if hasattr(self.model, 'hf_device_map'):
                        print(f"[SCG_LocalVLM]   Device map: {self.model.hf_device_map}")

                    # Check if model is actually quantized
                    is_quantized = False
                    for name, param in self.model.named_parameters():
                        if hasattr(param, 'quant_state'):
                            is_quantized = True
                            break

                    if is_quantized:
                        print(f"[SCG_LocalVLM]   Quantization: VERIFIED (layers are quantized)")
                    else:
                        print(f"[SCG_LocalVLM]   WARNING: Quantization requested but layers not quantized!")
                        
            except Exception as e:
                print(f"[SCG_LocalVLM] Error loading model: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
                # Try fallback without attention implementation
                if "attn_implementation" in load_kwargs:
                    print(f"[SCG_LocalVLM] Retrying without {attention_mode} attention...")
                    load_kwargs.pop("attn_implementation", None)
                    
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
                    print("[SCG_LocalVLM] Model loaded with fallback attention")
                else:
                    raise

        processed_video_path = None
        result = None

        with torch.inference_mode():
            # Build user content list - images first (in order), then text
            user_content = []

            # Process images in order: image1, image2, image3, image4
            images = [image1, image2, image3, image4]
            image_count = 0
            img_convert_start = time.time()
            for idx, img in enumerate(images, start=1):
                if img is not None:
                    print(f"Processing image{idx}")
                    pil_image = tensor_to_pil(img)
                    user_content.append({
                        "type": "image",
                        "image": pil_image,
                    })
                    image_count += 1
            img_convert_time = time.time() - img_convert_start

            if image_count > 0:
                print(f"Total images added: {image_count}")
                print(f"[SCG_LocalVLM] Image conversion time: {img_convert_time:.3f}s")

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

                # Time vision info processing
                vision_start = time.time()
                image_inputs, video_inputs = process_vision_info(messages)
                vision_time = time.time() - vision_start
                print(f"[SCG_LocalVLM] Vision info processing time: {vision_time:.3f}s")

                # Time processor (image embedding generation) - this is often the slowest part
                processor_start = time.time()
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
                processor_time = time.time() - processor_start
                print(f"[SCG_LocalVLM] Processor (embedding) time: {processor_time:.3f}s")

                # CRITICAL: Verify inputs are actually on GPU
                print(f"[SCG_LocalVLM] Verifying input devices...")
                print(f"[SCG_LocalVLM]   input_ids device: {inputs.input_ids.device}")
                if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                    print(f"[SCG_LocalVLM]   pixel_values device: {inputs.pixel_values.device}")
                    print(f"[SCG_LocalVLM]   pixel_values shape: {inputs.pixel_values.shape}")
                if hasattr(inputs, 'image_grid_thw') and inputs.image_grid_thw is not None:
                    print(f"[SCG_LocalVLM]   image_grid_thw device: {inputs.image_grid_thw.device}")

                # Show input context size (important for understanding generation speed)
                input_token_count = inputs.input_ids.shape[1]
                print(f"[SCG_LocalVLM] Input context: {input_token_count} tokens (text + vision)")

                # Timing check - see if there's a delay before generation
                print(f"[SCG_LocalVLM] Building generation kwargs...")
                kwargs_start = time.time()
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "use_cache": True,  # CRITICAL: Enable KV cache for fast generation
                    "num_beams": 1,  # Greedy/sampling is faster than beam search
                }

                # Add pad_token_id to prevent warnings
                if hasattr(self.processor, 'tokenizer'):
                    if hasattr(self.processor.tokenizer, 'pad_token_id') and self.processor.tokenizer.pad_token_id is not None:
                        generation_kwargs["pad_token_id"] = self.processor.tokenizer.pad_token_id
                    elif hasattr(self.processor.tokenizer, 'eos_token_id') and self.processor.tokenizer.eos_token_id is not None:
                        generation_kwargs["pad_token_id"] = self.processor.tokenizer.eos_token_id

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

                kwargs_time = time.time() - kwargs_start
                print(f"[SCG_LocalVLM] Generation kwargs built in {kwargs_time:.3f}s")

                # Check GPU memory before generation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_mem_before = torch.cuda.memory_allocated(0) / 1024**3

                # CRITICAL: Force CUDA sync and clear cache before generation to avoid random slowness
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                print(f"[SCG_LocalVLM] Starting generation with {input_token_count} input tokens...")
                gen_start = time.time()
                generated_ids = self.model.generate(**inputs, **generation_kwargs)
                gen_time = time.time() - gen_start

                # Check GPU memory after generation
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    gpu_mem_after = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_mem_peak = torch.cuda.max_memory_allocated(0) / 1024**3
                    print(f"[SCG_LocalVLM] GPU Memory during generation: {gpu_mem_before:.2f}GB → {gpu_mem_after:.2f}GB (peak: {gpu_mem_peak:.2f}GB)")
                    torch.cuda.reset_peak_memory_stats(0)

                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                # Calculate tokens/sec
                if len(generated_ids_trimmed) > 0:
                    tokens_generated = len(generated_ids_trimmed[0])
                    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
                    print(f"[SCG_LocalVLM] Generation Performance:")
                    print(f"[SCG_LocalVLM]   Tokens: {tokens_generated}")
                    print(f"[SCG_LocalVLM]   Time: {gen_time:.2f}s")
                    print(f"[SCG_LocalVLM]   Speed: {tokens_per_sec:.2f} tokens/sec")

                    # Performance benchmarks for RTX 5090 + Qwen3-VL-4B
                    # Based on actual testing - see context.md
                    expected_speeds = {
                        "none": {"min": 15.0, "typical": 16.5, "attn": "sdpa or flash_attention_2"},
                        "4bit": {"min": 10.0, "typical": 50.0, "attn": "eager (quantization doesn't work well with FA2)"},
                        "8bit": {"min": 8.0, "typical": 40.0, "attn": "eager"},
                    }

                    # Check if performance is below expected
                    if quantization in expected_speeds:
                        expected = expected_speeds[quantization]
                        if tokens_per_sec < expected["min"]:
                            print(f"[SCG_LocalVLM] ⚠️  WARNING: Very slow generation speed!")
                            print(f"[SCG_LocalVLM]   Expected: {expected['typical']:.1f} tok/s")
                            print(f"[SCG_LocalVLM]   Got: {tokens_per_sec:.2f} tok/s ({tokens_per_sec/expected['typical']*100:.1f}% of expected)")
                            print(f"[SCG_LocalVLM]   Recommended attention: {expected['attn']}")
                            print(f"[SCG_LocalVLM] Possible causes:")
                            print(f"[SCG_LocalVLM]   1. Model loaded on CPU instead of GPU (check warnings above)")
                            print(f"[SCG_LocalVLM]   2. Wrong attention mode (try '{expected['attn']}')")
                            print(f"[SCG_LocalVLM]   3. Other processes using GPU (check nvidia-smi)")
                            if quantization == "none":
                                print(f"[SCG_LocalVLM]   4. Try 'sdpa' instead of 'flash_attention_2' (often faster)")
                            elif quantization == "4bit":
                                print(f"[SCG_LocalVLM]   4. Quantization kernels issue (pip install bitsandbytes --upgrade)")

                    # Show timing breakdown if images were processed
                    if image_count > 0:
                        total_prep_time = img_convert_time + vision_time + processor_time
                        print(f"[SCG_LocalVLM] Timing Breakdown:")
                        print(f"[SCG_LocalVLM]   Image conversion: {img_convert_time:.3f}s")
                        print(f"[SCG_LocalVLM]   Vision processing: {vision_time:.3f}s")
                        print(f"[SCG_LocalVLM]   Image embedding:  {processor_time:.3f}s")
                        print(f"[SCG_LocalVLM]   Text generation:  {gen_time:.3f}s")
                        print(f"[SCG_LocalVLM]   Total prep time:  {total_prep_time:.3f}s")
                        if processor_time > 1.0:
                            print(f"[SCG_LocalVLM] TIP: Image embedding is slow. Try:")
                            print(f"[SCG_LocalVLM]   - Use 'low' or 'medium' image_resolution for faster processing")
                            print(f"[SCG_LocalVLM]   - Current: {image_resolution}")

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
    """
    Qwen Text-Only Language Model Node for ComfyUI

    Supports Qwen2.5 and Qwen3 text generation models with:
    - Multiple quantization options (none, 4bit, 8bit)
    - Advanced generation parameters
    - Flash Attention 2 optimization
    - System and user prompt support

    ComfyUI v2/v3 compatible.
    """

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
                "attention_mode": (
                    ["auto", "flash_attention_2", "sdpa", "eager"],
                    {"default": "auto"},
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
        attention_mode,
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
            load_start = time.time()
            
            # Check quantization support if using quantization
            if quantization != "none":
                quant_ok, quant_msg = _check_quantization_support()
                if not quant_ok:
                    print(f"[SCG_LocalVLM] WARNING: {quant_msg}")
                    print("[SCG_LocalVLM] Falling back to unquantized loading")
                    quantization = "none"
                else:
                    print(f"[SCG_LocalVLM] {quant_msg}")
            
            # Determine compute dtype
            compute_dtype = torch.bfloat16 if self.bf16_support else torch.float16
            print(f"[SCG_LocalVLM] Using compute dtype: {compute_dtype}")

            # Build load_kwargs - SIMPLIFIED for maximum performance
            load_kwargs = {
                "torch_dtype": compute_dtype,
            }
            
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
                print(f"[SCG_LocalVLM] Loading with 4-bit quantization")
                
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=compute_dtype,
                )
                load_kwargs["quantization_config"] = quantization_config
                load_kwargs["device_map"] = "auto"
                print(f"[SCG_LocalVLM] Loading with 8-bit quantization")
                
            else:
                # Non-quantized: NO device_map - use native PyTorch loading
                print("[SCG_LocalVLM] Loading without quantization (native PyTorch)")

            # Handle attention mode - simplified
            if attention_mode == "flash_attention_2" and _FA2_COMPATIBLE:
                try:
                    import flash_attn
                    load_kwargs["attn_implementation"] = "flash_attention_2"
                    attention_used = "flash_attention_2"
                except ImportError:
                    load_kwargs["attn_implementation"] = "sdpa"
                    attention_used = "sdpa"
            elif attention_mode == "sdpa" or (attention_mode == "flash_attention_2" and not _FA2_COMPATIBLE):
                load_kwargs["attn_implementation"] = "sdpa"
                attention_used = "sdpa"
            elif attention_mode == "eager":
                load_kwargs["attn_implementation"] = "eager"
                attention_used = "eager"
            else:
                load_kwargs["attn_implementation"] = "sdpa"
                attention_used = "sdpa"
            
            print(f"[SCG_LocalVLM] Using {attention_used} attention")

            # Load the model
            print(f"[SCG_LocalVLM] Loading model from {self.model_checkpoint}")
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_checkpoint,
                    **load_kwargs,
                )
                
                load_time = time.time() - load_start
                
                # For non-quantized: move to GPU
                if quantization == "none":
                    self.model = self.model.cuda()
                
                self.model.eval()
                
                # Print comprehensive model info
                print(f"[SCG_LocalVLM] Model loaded successfully in {load_time:.2f}s")
                print(f"[SCG_LocalVLM]   Device: {self.model.device}")
                print(f"[SCG_LocalVLM]   Dtype: {self.model.dtype}")
                print(f"[SCG_LocalVLM]   Attention: {attention_used}")

                # Verify actual attention implementation
                if hasattr(self.model.config, '_attn_implementation'):
                    actual_attn = self.model.config._attn_implementation
                    print(f"[SCG_LocalVLM]   Attention (verified): {actual_attn}")
                    if actual_attn != attention_used:
                        print(f"[SCG_LocalVLM]   WARNING: Requested {attention_used} but got {actual_attn}!")

                # GPU memory diagnostics
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated(0) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(0) / 1024**3
                    print(f"[SCG_LocalVLM]   GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")

                # Verify model is on GPU for non-quantized
                if quantization == "none":
                    # Check first parameter device
                    first_param = next(self.model.parameters())
                    if not first_param.is_cuda:
                        print(f"[SCG_LocalVLM]   WARNING: Model is on {first_param.device}, not CUDA!")
                        print(f"[SCG_LocalVLM]   This will cause VERY slow inference!")

                # Show quantization info
                if quantization != "none":
                    print(f"[SCG_LocalVLM]   Quantization: {quantization}")
                    if hasattr(self.model, 'hf_device_map'):
                        print(f"[SCG_LocalVLM]   Device map: {self.model.hf_device_map}")

                    # Check if model is actually quantized
                    is_quantized = False
                    for name, param in self.model.named_parameters():
                        if hasattr(param, 'quant_state'):
                            is_quantized = True
                            break

                    if is_quantized:
                        print(f"[SCG_LocalVLM]   Quantization: VERIFIED (layers are quantized)")
                    else:
                        print(f"[SCG_LocalVLM]   WARNING: Quantization requested but layers not quantized!")
                        
            except Exception as e:
                print(f"[SCG_LocalVLM] Error loading model: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
                # Try fallback without attention implementation
                if "attn_implementation" in load_kwargs:
                    print(f"[SCG_LocalVLM] Retrying without {attention_mode} attention...")
                    load_kwargs.pop("attn_implementation", None)
                    
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_checkpoint,
                        **load_kwargs,
                    )
                    print("[SCG_LocalVLM] Model loaded with fallback attention")
                else:
                    raise

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
                    "use_cache": True,  # CRITICAL: Enable KV cache
                }

                # Add pad_token_id to prevent warnings and improve batching
                if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                    generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
                elif hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                    generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

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

                # Start timing
                import time
                gen_start = time.time()

                generated_ids = self.model.generate(**inputs, **generation_kwargs)

                # Calculate performance
                gen_time = time.time() - gen_start

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                # Calculate tokens/sec
                if len(generated_ids_trimmed) > 0:
                    tokens_generated = len(generated_ids_trimmed[0])
                    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
                    print(f"[SCG_LocalVLM] Generation Performance:")
                    print(f"[SCG_LocalVLM]   Tokens: {tokens_generated}")
                    print(f"[SCG_LocalVLM]   Time: {gen_time:.2f}s")
                    print(f"[SCG_LocalVLM]   Speed: {tokens_per_sec:.2f} tokens/sec")

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
