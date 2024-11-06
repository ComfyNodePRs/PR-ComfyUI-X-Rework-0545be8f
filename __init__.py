from .src.py.utils.ClearNode import ClearNode
from .src.py.training.LoraTrainer import LoraTrainer

NODE_CLASS_MAPPINGS = {
    "ClearNode" : ClearNode,
    "LoraTrainer" : LoraTrainer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearNode" : "ClearNode",
    "LoraTrainer" : "LoraTrainer",
}

WEB_DIRECTORY = "./src/web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']