from .src.py.utils.ClearNode import ClearNode
from .src.py.img.LoadImage import LoadImageURL
from .src.py.img.UploadImage import UploadImage
from .src.py.sampler.XSampler import XSampler
from .src.py.training.LoraTrainer import LoraTrainer

def LoadDevModels():
    pass

NODE_CLASS_MAPPINGS = {
    "ClearNode" : ClearNode,
    "LoadImageURL" : LoadImageURL,
    "UploadImage" : UploadImage,
    #"XSampler" : XSampler
    #"LoraTrainer" : LoraTrainer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearNode" : "ClearNode",
    "LoadImageURL" : "LoadImage (URL)",
    "UploadImage" : "UploadImage (DISCORD)",
    #"XSampler" : "XSampler",
    #"LoraTrainer" : "LoraTrainer",
}

WEB_DIRECTORY = "./src/web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']