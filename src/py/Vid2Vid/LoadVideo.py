import os
import folder_paths

input_path = folder_paths.get_input_directory()
out_path = folder_paths.get_output_directory()

class LoadVideo:
    @classmethod
    def INPUT_TYPES(s):
        files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f)) and f.split('.')[-1] in ["mp4", "webm","mkv","avi"]]
        return {"required":{
            "video":(files,),
        }}

    ## Info ##
    CATEGORY = "comfy-rework/vid2vid"
    DESCRIPTION = "Loads a Video File."
    OUTPUT_NODE = False

    ## Func ##
    RETURN_TYPES = ("VIDEO",)
    FUNCTION = "load_video"

    def load_video(self, video):
        video_path = os.path.join(input_path,video)
        return (video_path,)