from .Lora import LoRA
from .Dataset import ImageDataset

from safetensors.torch import save_file
from torch import nn

# Define a function to apply LoRA to the UNet model
def apply_lora_to_unet(unet, lora_rank=4):
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            lora = LoRA(module.in_features, lora_rank)
            module.add_module("lora", lora)

# Define a function to save only the LoRA weights in Safetensors format
def safestore_lora(unet, path="safestore_lora.safetensors"):
    lora_state_dict = {}
    for name, module in unet.named_modules():
        if hasattr(module, "lora"):
            # Save only the LoRA parameters
            lora_state_dict[f"{name}.down_proj.weight"] = module.lora.down_proj.weight.data.cpu()
            lora_state_dict[f"{name}.up_proj.weight"] = module.lora.up_proj.weight.data.cpu()

    # Save the weights using Safetensors format
    save_file(lora_state_dict, path)
    print(f"LoRA weights saved to {path}")
    
def auto_detect_batch_size(dataset_path, max_batch_size=32):
    # Count the number of images in the dataset
    dataset = ImageDataset(dataset_path)
    num_images = len(dataset)
    
    # Automatically set the batch size based on the number of images
    if num_images <= max_batch_size:
        batch_size = num_images  # Use all images in a single batch if it's small
    else:
        batch_size = min(max_batch_size, num_images // 10)  # Divide dataset by 10 if too large
    
    print(f"Total images: {num_images}, Setting batch size to: {batch_size}")
    return batch_size