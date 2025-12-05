# ComfyUI_QwenVL Optimization Project - Context

**Last Updated:** 2025-12-05
**Branch:** `claude/implement-fixes-enhancements-01Cxt5wVVQY4yqXMYnr3Q8ax`

## Project Overview

Implementing comprehensive fixes and enhancements to the ComfyUI_QwenVL node pack, including critical bug fixes, missing generation parameters, performance optimizations, and ensuring ComfyUI node v3 compliance.

## Current Status

### Phase 1: Critical Fixes ✅ COMPLETED
- [x] Fix 1.1: Non-quantized model loading (added `low_cpu_mem_usage=True`)
- [x] Fix 1.2: 8bit quantization compute_dtype parameter (added `bnb_8bit_compute_dtype`)
- [x] Fix 1.3: Hardcoded CUDA device calls (changed `.to("cuda")` -> `.to(self.model.device)`)
- [x] Fix 1.4: Temperature parameter bug (removed from batch_decode())

### Phase 2: Core Generation Parameters ✅ COMPLETED
- [x] Add `do_sample` parameter (default: True)
- [x] Add `repetition_penalty` parameter (default: 1.0, range: 1.0-2.0)
- [x] Add `top_p` parameter (default: 0.9, range: 0.0-1.0)
- [x] Add `top_k` parameter (default: 50, range: 0-200)
- [x] Add `min_p` parameter (default: 0.0, range: 0.0-1.0)
- [x] Update generate() calls to use new parameters with proper conditional logic
- [x] Expand temperature range (0-2 instead of 0-1) and improve step size (0.05)

### Phase 3: Performance Optimizations ✅ COMPLETED
- [x] Add Flash Attention 2 support (optional, with graceful fallback)
- [x] Switch from `torch.no_grad()` to `torch.inference_mode()`
- [ ] Refactor duplicate model loading logic into shared functions (DEFERRED - code working well)

### Phase 4: Quality Improvements ✅ COMPLETED
- [x] Improve error messages (specific CUDA OOM handling with actionable suggestions)
- [x] Add detailed exception logging with traceback for debugging
- [ ] Add performance logging (tokens/sec) (DEFERRED - can be added later)
- [ ] Add parameter validation (DEFERRED - ComfyUI handles basic validation)

### Phase 5: ComfyUI v3 Compliance ✅ COMPLETED
- [x] Research ComfyUI node v3 requirements
- [x] Add RETURN_NAMES for better output clarity
- [x] Verify standard node structure compliance

## Code Locations

### QwenVL Class (Vision-Language)
- **INPUT_TYPES**: Lines 208-246
- **Model Loading**: Lines 303-337
- **Inference**: Lines 252-426
- **CUDA device call**: Line 403
- **batch_decode call**: Line 409-414

### Qwen Class (Text-only)
- **INPUT_TYPES**: Lines 449-480
- **Model Loading**: Lines 524-548
- **Inference**: Lines 486-581
- **CUDA device call**: Line 562
- **batch_decode call**: Line 569-574

## Issues Identified

### Critical Bugs
1. **Non-quantized models load to CPU first** (Lines 319-337, 540-548)
   - Missing `low_cpu_mem_usage=True` in load_kwargs
   - Causes unnecessary RAM usage and slow loading

2. **8bit quantization incomplete** (Lines 315-318, 536-539)
   - Missing `bnb_8bit_compute_dtype=compute_dtype`
   - May cause performance issues or errors

3. **Hardcoded CUDA calls fail on CPU** (Lines 403, 562)
   - `.to("cuda")` should be `.to(self.model.device)`
   - Breaks when running on CPU

4. **Temperature parameter bug** (Lines 413, 573)
   - `batch_decode()` doesn't accept temperature parameter
   - Will cause runtime error or be silently ignored

### Missing Features
5. **No sampling control**
   - `do_sample` not exposed (temperature set but sampling not explicitly enabled)
   - Missing `top_p`, `top_k`, `min_p` for quality control
   - Missing `repetition_penalty` for preventing repetitive output

## Research Notes

### Qwen VL 3 Specifics
- Uses `Qwen3VLForConditionalGeneration` class
- Already implemented in code with automatic detection (line 324)
- Custom models supported via `custom_models.json`

### BitsAndBytes Quantization
- 4bit config is correct (lines 309-314, 530-535)
- 8bit missing compute_dtype (needs fixing)
- Both use `device_map="auto"` correctly

### ComfyUI Node Architecture
- Uses standard INPUT_TYPES/RETURN_TYPES/FUNCTION pattern
- Currently registered in __init__.py
- Need to research v3 specific requirements

## Testing Plan

After implementing fixes:
1. Test `quantization="none"` - verify GPU loading (check nvidia-smi)
2. Test `quantization="4bit"` - verify functional
3. Test `quantization="8bit"` - verify functional with new compute_dtype
4. Test CPU fallback - verify no CUDA errors
5. Test new generation parameters affect output quality
6. Test temperature values (0.0, 0.7, 1.0)
7. Test repetition_penalty reduces repetition

## Changes Summary

### Critical Bug Fixes
1. **Model Loading (Both Classes)**
   - **CRITICAL FIX**: Added `torch_dtype=compute_dtype` to ALL quantization paths (4bit, 8bit, none)
     - Without this, quantized models default to float32 causing massive CPU RAM usage
     - This was the root cause of poor GPU utilization and CPU bottleneck
   - Added `low_cpu_mem_usage=True` for all loading paths
   - Added `bnb_8bit_compute_dtype=compute_dtype` to 8bit quantization config
   - Restructured loading logic to use `load_kwargs` dict for cleaner code

2. **Device Compatibility**
   - Fixed hardcoded `.to("cuda")` calls → `.to(self.model.device)` (QwenVL line 413, Qwen line 648)
   - Ensures compatibility when running on CPU

3. **Temperature Bug**
   - Removed `temperature` parameter from `batch_decode()` calls (it doesn't accept this parameter)
   - Temperature is now only used in `generate()` where it belongs

### New Generation Parameters (Both Classes)
- `do_sample` (BOOLEAN, default: True) - Enable/disable sampling
- `top_p` (FLOAT, 0.0-1.0, default: 0.9) - Nucleus sampling
- `top_k` (INT, 0-200, default: 50) - Top-k sampling
- `min_p` (FLOAT, 0.0-1.0, default: 0.0) - Minimum probability threshold
- `repetition_penalty` (FLOAT, 1.0-2.0, default: 1.0) - Prevent repetition
- Expanded `temperature` range to 0-2 with 0.05 step size

### Performance Optimizations
1. **Flash Attention 2** ✅ ENHANCED
   - Proper ImportError detection (checks for flash_attn package)
   - Graceful fallback with automatic retry if Flash Attention fails
   - Verification logging shows which attention implementation actually loaded
   - Significantly improves attention computation speed when available

2. **Inference Mode**
   - Switched from `torch.no_grad()` to `torch.inference_mode()`
   - Provides better performance by more aggressively disabling gradient tracking

3. **KV Cache** ✅ CRITICAL
   - Added `use_cache=True` to all generate() calls (both classes)
   - Essential for quantized model performance (especially 4bit/8bit)
   - Without KV cache, every token regenerates full attention = major slowdown
   - Expected improvement: 4bit should now be ~equal or faster than unquantized

4. **4bit Quantization Optimization** ✅ NEW
   - Added `llm_int8_threshold=6.0` to BitsAndBytesConfig for 4bit
   - Keeps outlier features in FP16 for better performance
   - Critical for achieving fast 4bit inference

5. **Tensor Optimization** ✅ NEW
   - Optimized tensor_to_pil() to perform operations on GPU before CPU transfer
   - Uses `.clamp()` and `.byte()` on GPU instead of CPU numpy operations
   - Reduces unnecessary CPU-GPU memory transfers

6. **Pad Token Handling** ✅ NEW
   - Properly sets pad_token_id in generation_kwargs
   - Prevents warnings and improves batching efficiency
   - Falls back to eos_token_id if pad_token_id not available

7. **Performance Logging** ✅ NEW
   - Real-time tokens/sec measurement during generation
   - Model loading time tracking
   - Comprehensive device/dtype/quantization verification logging
   - Helps diagnose performance issues quickly

### Error Handling Improvements
- Specific `torch.cuda.OutOfMemoryError` handling with actionable suggestions
- Detailed exception logging with traceback for debugging
- User-friendly error messages with `[SCG_LocalVLM]` prefix

### ComfyUI v3 Compliance ✅ ENHANCED
- Added `RETURN_NAMES = ("text",)` to both node classes
- Verified standard node structure (INPUT_TYPES, RETURN_TYPES, FUNCTION, CATEGORY)
- Enhanced __init__.py with:
  - Module docstring explaining features and compatibility
  - `__version__` export for v3 introspection
  - `__all__` export list for proper module interface
  - Better display names for nodes
- Added comprehensive docstrings to both node classes
- Maintains full backwards compatibility with ComfyUI v2
- Ready for ComfyUI v3 public API (stateless execution model compatible)

## Testing Checklist

### Critical Functionality
- [x] Code compiles without syntax errors
- [ ] `quantization="none"` loads model directly to GPU (check nvidia-smi)
- [ ] `quantization="4bit"` works correctly
- [ ] `quantization="8bit"` works correctly with new compute_dtype
- [ ] CPU fallback works without CUDA errors
- [ ] All new parameters accessible in ComfyUI interface

### Generation Quality
- [ ] `do_sample=False` produces deterministic output
- [ ] `do_sample=True` produces varied output
- [ ] `temperature` affects output randomness (test 0.0, 0.7, 1.5)
- [ ] `repetition_penalty` reduces repetitive text
- [ ] `top_p` and `top_k` affect output quality
- [ ] `min_p` works as expected

### Error Handling
- [ ] CUDA OOM error provides helpful suggestions
- [ ] General errors show detailed traceback in console
- [ ] Model loading errors are handled gracefully

## Notes

- Video-related improvements deferred (as requested)
- Code refactoring into shared functions deferred (current structure is clear and working)
- All changes are backwards compatible
- Temperature now properly used only in generation, not decoding

## Performance Optimization Session (2025-12-05)

### Final Results (QwenVL, Qwen3-VL-4B on RTX 5090)
| Mode | Attention | Speed | Status |
|------|-----------|-------|--------|
| Non-quantized | SDPA | 16.8 tok/s | ✅ Target met |
| Non-quantized | FA2 | 16.3 tok/s | ✅ Works on Blackwell |
| Non-quantized | Eager | 12.8 tok/s | ✅ Working |
| 4-bit | FA2 | 11.8 tok/s | ⚠️ Slower than expected |
| 4-bit | SDPA | 5.0 tok/s | ⚠️ Very slow |
| 4-bit | Eager | 12.7 tok/s | ✅ Best for 4-bit |

### Key Changes Made
1. **FA2 Detection** - Changed from blocker to hint (`_FA2_OPTIMIZED`), future-proof
2. **device_map Optimization** - `{"":0}` for single GPU quantized (avoids Accelerate overhead)
3. **Direct GPU Loading** - `{"":"cuda:0"}` for non-quantized (no CPU→GPU step)
4. **Removed .cuda() call** - device_map handles GPU placement

### Remaining Issues (External)
- **4-bit slow on Blackwell (SM 120)** - bitsandbytes kernels not optimized for RTX 50 series
- Workaround: Use eager attention for 4-bit, or use non-quantized
- Will likely be fixed in future bitsandbytes release

### Qwen Text Model
- Same optimizations NOT yet applied (edit tool issues during session)
- TODO: Apply same device_map and FA2 changes to Qwen text class
