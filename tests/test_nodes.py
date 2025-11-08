import atexit
import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - handled by skip below
    torch = None


if torch is not None:
    _TEMP_DIR = tempfile.TemporaryDirectory()
    atexit.register(_TEMP_DIR.cleanup)
    folder_paths_module = types.ModuleType("folder_paths")
    folder_paths_module.models_dir = _TEMP_DIR.name
    sys.modules["folder_paths"] = folder_paths_module

    nodes = importlib.import_module("nodes")
    nodes = importlib.reload(nodes)

    class DummyInputs(dict):
        def __init__(self):
            tensor = torch.tensor([[0, 1]])
            super().__init__({"input_ids": tensor})
            self.input_ids = tensor

        def to(self, device):
            self.device = device
            return self

    class DummyProcessor:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "template"

        def __call__(self, **kwargs):
            return DummyInputs()

        def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False, temperature=0.0):
            return ["decoded vision output"]

    class DummyTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "tokenized"

        def __call__(self, texts, return_tensors="pt"):
            return DummyInputs()

        def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False, temperature=0.0):
            return ["decoded text output"]

    class DummyVLModel:
        def __init__(self):
            self.to_calls = []

        def to(self, device):
            self.to_calls.append(device)
            return self

        def generate(self, **kwargs):
            return torch.tensor([[0, 1, 2, 3]])

    class DummyTextModel(DummyVLModel):
        pass

    class NodesTestCase(unittest.TestCase):
        def setUp(self):
            self.llm_dir = os.path.join(nodes.folder_paths.models_dir, "LLM")
            os.makedirs(self.llm_dir, exist_ok=True)

        def _ensure_checkpoint_path(self, model_name):
            checkpoint_dir = os.path.join(self.llm_dir, model_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            return checkpoint_dir

        @patch("nodes.process_vision_info", return_value=([None], None))
        @patch("nodes.Qwen3VLForConditionalGeneration.from_pretrained", return_value=DummyVLModel())
        @patch("nodes.AutoProcessor.from_pretrained", return_value=DummyProcessor())
        @patch("nodes._clear_cuda_memory")
        def test_qwenvl_unloads_models_after_run(self, clear_mock, _processor_mock, _model_mock, _vision_mock):
            self._ensure_checkpoint_path("Qwen3-VL-4B-Instruct")
            node = nodes.QwenVL()
            image = torch.zeros((1, 1, 1, 3))

            result = node.inference(
                text="hi",
                model="Qwen3-VL-4B-Instruct",
                quantization="none",
                keep_model_loaded=False,
                temperature=0.7,
                max_new_tokens=10,
                seed=-1,
                image=image,
                video_path="",
            )

            self.assertEqual(result, ["decoded vision output"])
            self.assertIsNone(node.model)
            self.assertIsNone(node.processor)
            clear_mock.assert_called_once()

        @patch("nodes.AutoModelForCausalLM.from_pretrained", return_value=DummyTextModel())
        @patch("nodes.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
        @patch("nodes._clear_cuda_memory")
        def test_qwen_text_node_unloads_when_not_kept(self, clear_mock, _tokenizer_mock, _model_mock):
            self._ensure_checkpoint_path("Qwen3-4B-Instruct-2507")
            node = nodes.Qwen()

            result = node.inference(
                system="sys",
                prompt="hi",
                model="Qwen3-4B-Instruct-2507",
                quantization="none",
                keep_model_loaded=False,
                temperature=0.7,
                max_new_tokens=10,
                seed=-1,
            )

            self.assertEqual(result, ["decoded text output"])
            self.assertIsNone(node.model)
            self.assertIsNone(node.tokenizer)
            clear_mock.assert_called_once()

        @patch("nodes.AutoModelForCausalLM.from_pretrained", return_value=DummyTextModel())
        @patch("nodes.AutoTokenizer.from_pretrained", return_value=DummyTokenizer())
        @patch("nodes._clear_cuda_memory")
        def test_qwen_text_node_keeps_model_when_requested(self, clear_mock, _tokenizer_mock, _model_mock):
            self._ensure_checkpoint_path("Qwen3-4B-Instruct-2507")
            node = nodes.Qwen()

            node.inference(
                system="sys",
                prompt="hi",
                model="Qwen3-4B-Instruct-2507",
                quantization="none",
                keep_model_loaded=True,
                temperature=0.7,
                max_new_tokens=10,
                seed=-1,
            )

            self.assertIsNotNone(node.model)
            self.assertIsNotNone(node.tokenizer)
            clear_mock.assert_not_called()


else:

    class NodesTestCase(unittest.TestCase):
        def test_pytorch_dependency_required(self):
            self.skipTest("PyTorch is not installed; skipping nodes tests.")


if __name__ == "__main__":
    unittest.main()
