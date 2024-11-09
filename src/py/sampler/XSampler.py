import comfy
import torch
import random
import latent_preview

def x_sampler(model, clip, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False):
    latent_image = latent["samples"]
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    seed = random.randrange(1, 10000000)
    steps = 25
    cfg = 7.0

    pos_tokens = clip.tokenize(positive)
    pos_output = clip.encode_from_tokens(pos_tokens, return_pooled=True, return_dict=True)
    pos_cond = pos_output.pop("cond")
    
    neg_tokens = clip.tokenize(negative)
    neg_output = clip.encode_from_tokens(neg_tokens, return_pooled=True, return_dict=True)
    neg_cond = neg_output.pop("cond")

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    callback = latent_preview.prepare_callback(model, steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, pos_cond, neg_cond, latent_image,
                                  denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                  force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    out = latent.clone()    
    out["samples"] = samples
    return (out, )

class XSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "clip": ("CLIP",), 
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "The algorithm used when sampling, this can affect the quality, speed, and style of the generated output."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "positive": ("STRING", {"multiline": True ,"default": "Positiv Text."}),
                "negative": ("STRING", {"multiline": True ,"default": "Negativ Text."}),
                "latent_image": ("LATENT", {"tooltip": "The latent image to denoise."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of denoising applied, lower values will maintain the structure of the initial image allowing for image to image sampling."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The denoised latent.",)
    FUNCTION = "sample"

    CATEGORY = 'rework-x/sampler'

    def sample(self, model, clip, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return x_sampler(model, clip, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)