from .nodes import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Qwen2.5VL": QwenVL,
    "Qwen2.5": Qwen,
    "QwenVL": QwenVL,
    "Qwen": Qwen
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2.5VL": "Qwen2.5VL",
    "Qwen2.5": "Qwen2.5",
    "QwenVL": "QwenVL",
    "Qwen": "Qwen",
}
