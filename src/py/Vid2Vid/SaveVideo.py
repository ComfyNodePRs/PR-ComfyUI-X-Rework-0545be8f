import os
import cv2
import numpy as np
import torch
import folder_paths

class SaveVideo:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.codec = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4 codec
        self.fps = 30  # Default frames per second

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("VIDEO", {"tooltip": "The frames to save as video."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. Can include formatting info like %date:yyyy-MM-dd%."}),
                "fps": ("INT", {"default": 30, "tooltip": "Frames per second for the output video."})
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"

    OUTPUT_NODE = True

    CATEGORY = "video"
    DESCRIPTION = "Saves the input frames as a video to your ComfyUI output directory."

    def save_video(self, frames, filename_prefix="ComfyUI", fps=30, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        h, w = frames[0].shape[2], frames[0].shape[3]
        
        # Generate the output path and filename
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, w, h
        )
        
        # Define the output video file path
        video_file_path = os.path.join(full_output_folder, f"{filename}_{counter:05}.mp4")
        
        # Initialize the video writer
        out = cv2.VideoWriter(video_file_path, self.codec, fps, (w, h))
        
        for frame in frames:
            # Convert the tensor to a numpy array in BGR format for OpenCV
            frame = (255. * frame.cpu().numpy().transpose(1, 2, 0)).astype(np.uint8)  # (H, W, C) format
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
        
        return {
            "ui": {
                "videos": [{
                    "filename": video_file_path,
                    "subfolder": subfolder,
                    "type": self.type
                }]
            }
        }