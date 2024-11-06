import torch

class AnyType(str):
  """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""
  def __eq__(self, __value: object) -> bool:
    return True
  def __ne__(self, __value: object) -> bool:
    return False

def clear_memory():
    import gc
    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

any = AnyType("*")
class ClearNode:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anything": (any, {}),
                "clear_cache": ("BOOLEAN", {"default": True}),
                "clear_models": ("BOOLEAN", {"default": True}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "clear_vram"
    CATEGORY = 'comfy-rework/utils'
    OUTPUT_NODE = True

    def clear_vram(self, anything, clear_cache, clear_models):
        import torch.cuda
        import gc
        import comfy.model_management
        clear_memory()
        if clear_models:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
        return (None,)
