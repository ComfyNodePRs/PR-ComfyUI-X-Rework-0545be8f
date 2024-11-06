import os
import torch
from torch import nn
from torch.optim import AdamW
from diffusers import StableDiffusionPipeline, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
from datasets import load_dataset

class LoRA(nn.Module):
    def __init__(self, original_dim, rank):
        super(LoRA, self).__init__()
        self.rank = rank
        self.down_proj = nn.Linear(original_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, original_dim, bias=False)

    def forward(self, x):
        return self.up_proj(self.down_proj(x))

# Define a function to apply LoRA to the UNet model
def apply_lora_to_unet(unet, lora_rank=4):
    for name, module in unet.named_modules():
        if isinstance(module, nn.Linear):
            lora = LoRA(module.in_features, lora_rank)
            module.add_module("lora", lora)

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
                "model" : ("Model",),
                "lora_rank" : ("INT",),
                "output_name" : ("STRING",)
            },
            "optional": {
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "train_lora"
    CATEGORY = 'comfy-rework/training'
    OUTPUT_NODE = True

    def train_lora(self, dataset, epochs, steps_per_epoch, model, output_name, lora_rank=4, lr=5e-5):
        if not os.path.isdir(dataset): return
        else:
            # Load the model, tokenizer, and scheduler
            tokenizer = CLIPTokenizer.from_pretrained(model)
            text_encoder = CLIPTextModel.from_pretrained(model)
            unet = UNet2DConditionModel.from_pretrained(model)
            scheduler = DDPMScheduler.from_pretrained(model)

            # Apply LoRA layers to the UNet model
            apply_lora_to_unet(unet, lora_rank=lora_rank)
            
            # Load dataset
            dataset = load_dataset(dataset, split="train")

            # Define transformations for images
            preprocess = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])

            # Define optimizer
            optimizer = AdamW(unet.parameters(), lr=lr)

            # Training loop
            for epoch in range(epochs):
                for i, sample in enumerate(dataset):
                    # Process image and text
                    image = preprocess(sample["image"]).unsqueeze(0)
                    prompt = sample["text"]
                    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

                    # Move to device
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    image, unet, text_encoder = image.to(device), unet.to(device), text_encoder.to(device)

                    # Generate noise and compute loss
                    noise = torch.randn_like(image)
                    timesteps = torch.randint(0, scheduler.num_train_timesteps, (1,), device=device).long()
                    encoder_hidden_states = text_encoder(inputs.input_ids.to(device)).last_hidden_state

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