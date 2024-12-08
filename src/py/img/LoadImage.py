import os
import torch
import urllib.request
import numpy as np
import folder_paths
import node_helpers

from PIL import Image, ImageOps, ImageSequence

from ..ErrorHandler import ErrorHandler

input_dir = folder_paths.get_input_directory()

class LoadImageURL:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"url": ("STRING", {"default" : ""}),}
                }

    CATEGORY = 'x-rework/image'
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    
    def load_image(self, url):
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-Agent', 'Chrome')]
            urllib.request.install_opener(opener)
            
            original_filename = url.split("/")[-1]
            
            image_path = os.path.join(input_dir, original_filename)
            
            urllib.request.urlretrieve(url, image_path)

            # Load Img #        
            img = node_helpers.pillow(Image.open, image_path)
            
            output_images = []
            output_masks = []
            w, h = None, None

            excluded_formats = ['MPO']
            
            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")

                if len(output_images) == 0:
                    w = image.size[0]
                    h = image.size[1]
                
                if image.size[0] != w or image.size[1] != h:
                    continue
                
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                if 'A' in i.getbands():
                    mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask = 1. - torch.from_numpy(mask)
                else:
                    mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
                output_images.append(image)
                output_masks.append(mask.unsqueeze(0))

            if len(output_images) > 1 and img.format not in excluded_formats:
                output_image = torch.cat(output_images, dim=0)
                output_mask = torch.cat(output_masks, dim=0)
            else:
                output_image = output_images[0]
                output_mask = output_masks[0]

        except Exception as e:
            ErrorHandler().handle_error("image", f"Error loading image from {url}.")
            return (None, )

        return (output_image, output_mask)
