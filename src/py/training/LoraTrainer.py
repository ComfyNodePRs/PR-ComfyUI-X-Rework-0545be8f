import os
import pathlib
import torch
from torch import nn
from torch.optim import AdamW
from diffusers import UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
import folder_paths

from .trainer.Dataset import ImageDataset
from .trainer.Utils import apply_lora_to_unet, safestore_lora, auto_detect_batch_size, load_models

output_dir = folder_paths.get_output_directory()

class LoraTrainer:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_path": ("STRING", {"default" : ""}),
                "model_path": ("STRING", {"default" : ""}),
                "epochs": ("INT", {"default": 1, "min": 1, "max": 10000, "tooltip": "The number of epochs the lora gets trained."}),
                "base_model" : ("MODEL",),
                "output_name" : ("STRING", {"default" : "output_model"})
            },
            "optional": {
            }
        }

    CATEGORY = 'comfy-rework/training'

    RETURN_TYPES = ()
    FUNCTION = "train_lora"

    OUTPUT_NODE = True

    def train_lora(self, dataset_path, model_path, epochs, base_model, output_name, lr=5e-5):
        if not os.path.isdir(dataset_path): return
        else:
            models_dir = os.path.join(pathlib.Path().parent.parent.parent.parent.resolve() + "model")
            tokenizer, model = load_models(models_dir)
            text_encoder = CLIPTextModel.from_pretrained(model_path)
            unet = UNet2DConditionModel.from_pretrained(model_path)
            scheduler = DDPMScheduler.from_pretrained(model_path)
            
            batch_size = auto_detect_batch_size(dataset_path)

            # Apply LoRA layers to the UNet model
            apply_lora_to_unet(unet, lora_rank=4)
            
            # Load the dataset from a local folder
            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
            
            dataset = ImageDataset(dataset_path, transform=transform)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Define optimizer
            optimizer = AdamW(unet.parameters(), lr=lr)

            # Training loop
            for epoch in range(epochs):
                for i, image in enumerate(data_loader):
                    # Move to device
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    image = image.to(device)
                    
                    # Generate noise and compute loss
                    noise = torch.randn_like(image)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (image.size(0),), device=device).long()
                    encoder_hidden_states = text_encoder(input_ids=torch.zeros(image.size(0), 77, dtype=torch.long).to(device)).last_hidden_state

                    # Apply forward pass through UNet with LoRA
                    noisy_image = scheduler.add_noise(image, noise, timesteps)
                    model_output = unet(noisy_image, timesteps, encoder_hidden_states).sample

                    # Loss calculation
                    loss = nn.MSELoss()(model_output, noise)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if i % 10 == 0:
                        print(f"Epoch {epoch+1}/{epochs}, Step {i}, Loss: {loss.item()}")

            # Save LoRA weights
            safestore_lora(unet, path=os.path.join(output_dir + output_name + ".safetensors"))
            print("Training completed.")