import os

class LoraTrainer:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING",),
                "epochs": ("INT",),
                "steps_per_epoch": ("INT",),
                "output_name" : ("STRING",)
            },
            "optional": {
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "train_lora"
    CATEGORY = 'comfy-rework/training'
    OUTPUT_NODE = True

    def train_lora(self, dataset, epochs, steps_per_epoch, output_name, ):
        if not os.path.isdir(dataset): return
        else:
            #TODO: Implement training.
            return (None,)