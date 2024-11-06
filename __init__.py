from .py.utils.ClearNode import ClearNode
from .py.utils.ReQueue import ReQueue

NODE_CLASS_MAPPINGS = {
    "ClearNode" : ClearNode,
    "ReQueue" : ReQueue,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearNode" : "ClearNode",
    "ReQueue" : "ReQueue"
}

WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']