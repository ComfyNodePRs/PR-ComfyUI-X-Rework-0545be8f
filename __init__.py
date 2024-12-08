from .src.py.utils.ClearNode import ClearNode
from .src.py.img.LoadImage import LoadImageURL
from .src.py.img.UploadImage import UploadImage
from .src.py.img.SaveImage import XSave
from .src.py.sampler.XSampler import XSampler

def LoadDevModels():
    pass

NODE_CLASS_MAPPINGS = {
    "ClearNode" : ClearNode,
    "LoadImageURL" : LoadImageURL,
    "UploadImage" : UploadImage,
    "XSave" : XSave,
    #"XSampler" : XSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClearNode" : "ClearNode",
    "LoadImageURL" : "LoadImage (URL)",
    "UploadImage" : "UploadImage (DISCORD)",
    "XSave" : "XSave",    
    #"XSampler" : "XSampler",
}

WEB_DIRECTORY = "./src/web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']