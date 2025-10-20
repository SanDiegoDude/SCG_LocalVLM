# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI_QwenVL is a custom node extension for ComfyUI that implements Qwen2.5-VL and Qwen2.5 language models. It provides vision-language capabilities for text-based and single-image queries, along with text-only generation. The project integrates with ComfyUI's node-based workflow system.

## Architecture

The extension consists of two main node classes in `nodes.py`:

- **Qwen2VL**: Handles vision-language model inference with support for image and video inputs, now with Qwen3VL support
- **Qwen2**: Handles text-only language model inference

Both classes follow the same architectural pattern:
- Model loading with quantization support (none, 4bit, 8bit)
- Automatic model downloading to `ComfyUI/models/LLM/`
- Device detection and optimization (CUDA/CPU with bfloat16 support)
- Memory management with optional model persistence

### Key Components

- **Model Management**: Automatic downloading from HuggingFace, local caching, quantization
- **Input Processing**: Tensor-to-PIL conversion, video preprocessing with FFmpeg
- **Inference Pipeline**: Template application, tokenization, generation, decoding
- **Memory Optimization**: Conditional model unloading and CUDA cache cleanup

## Development Commands

### Installation
```bash
# Clone and install dependencies
git clone https://github.com/alexcong/ComfyUI_QwenVL.git
cd ComfyUI_QwenVL
pip install -r requirements.txt
```

### Dependencies Management
```bash
# Install/update dependencies
pip install -r requirements.txt

# Key dependencies include:
# - torch, torchvision, numpy, pillow
# - huggingface_hub, transformers, bitsandbytes, accelerate
# - qwen-vl-utils, optimum
# - transformers from git (main branch)
```

### Testing
```bash
# Test models can be loaded (requires ComfyUI environment)
# Load ComfyUI and verify nodes appear in "Comfyui_QwenVL" category
```

## Model Configuration

### Supported Models
- **Vision-Language**: Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct, Qwen3-VL-4B-Thinking, Qwen3-VL-8B-Thinking, SkyCaptioner-V1
- **Text-Only**: Qwen2.5-3B/7B/14B/32B-Instruct

### Model Location
Models are automatically downloaded to: `ComfyUI/models/LLM/`

### Model ID Formats
- **Qwen3 models**: Use `Qwen/{model_name}` format (e.g., `Qwen/Qwen3-VL-4B-Thinking`)
- **Qwen2.5 models**: Use `qwen/{model_name}` format (e.g., `qwen/Qwen2.5-VL-3B-Instruct`)
- **Skywork models**: Use `Skywork/{model_name}` format (e.g., `Skywork/SkyCaptioner-V1`)

### Quantization Options
- **none**: Full precision (bfloat16/float16 based on GPU capability)
- **4bit**: BitsAndBytes 4-bit quantization
- **8bit**: BitsAndBytes 8-bit quantization

## Code Structure

### File Organization
```
├── nodes.py              # Main node implementations (Qwen2VL, Qwen2) with Qwen3VL support
├── __init__.py          # Node class mappings for ComfyUI
├── pyproject.toml       # Project metadata and dependencies
├── requirements.txt     # Python dependencies
├── README.md           # User documentation
└── workflow/           # Example ComfyUI workflows
```

### Node Implementation Pattern
Both nodes follow ComfyUI's standard pattern:
- `INPUT_TYPES()`: Define input parameters and types
- `inference()`: Main processing method
- Model loading and caching in instance variables
- Device detection and optimization

## Important Implementation Details

### Qwen3VL Support
- **Model Detection**: Uses `model.startswith("Qwen3")` to identify Qwen3 models
- **Model Class**: Loads `Qwen3VLForConditionalGeneration` for Qwen3 models
- **Repository Format**: Uses `Qwen/{model_name}` format for HuggingFace repository
- **Backward Compatibility**: All existing Qwen2.5 and Skywork models continue to work unchanged

### Video Processing
- Uses FFmpeg for video frame extraction and resizing
- Processes videos to 1fps with max dimension 256px
- Creates temporary files in `/tmp/` with unique identifiers
- Automatically cleans up temporary files after inference

### Memory Management
- `keep_model_loaded` parameter controls model persistence
- Automatic CUDA cache cleanup when unloading models
- Device detection for optimal dtype selection (bfloat16 vs float16)

### Error Handling
- Graceful handling of empty prompts
- FFmpeg error handling for video processing
- Model inference exception catching

## Development Notes

### Adding New Models
1. Add model name to the model list in `INPUT_TYPES()`
2. Update model_id logic if needed (Qwen vs Skywork prefixes)
3. Test model downloading and inference

### ComfyUI Integration
- Nodes are registered in `__init__.py` with class and display name mappings
- Category is set to "Comfyui_QwenVL"
- Return type is always "STRING" for generated text

### Dependencies
The project requires specific versions of transformers and related libraries. Always use the provided requirements.txt for compatibility.