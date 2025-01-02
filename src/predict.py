import os
from typing import List
from utils import encode_video_to_base64
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# from diffusers.pipelines.stable_diffusion.safety_checker import (
#     StableDiffusionSafetyChecker,
# )


MODEL_ID = "THUDM/CogVideoX-5b"
MODEL_CACHE = "diffusers-cache"
# SAFETY_MODEL_ID = "CompVis/stable-diffusion-safety-checker"


class Predictor:
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")
        # safety_checker = StableDiffusionSafetyChecker.from_pretrained(
        #     SAFETY_MODEL_ID,
        #     cache_dir=MODEL_CACHE,
        #     local_files_only=True,
        # )
        # pipe = CogVideoXPipeline.from_pretrained(
        #     "THUDM/CogVideoX-5b",
        #     torch_dtype=torch.bfloat16
        # )
        self.pipe = CogVideoXPipeline.from_pretrained(
            MODEL_ID,
            safety_checker=None,
            torch_dtype=torch.bfloat16,
            # safety_checker=safety_checker,
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

        # self.pipe.enable_xformers_memory_efficient_attention()

    @torch.inference_mode()
    def predict(self, prompt, number_of_frames,num_inference_steps, guidance_scale,fps):
        if torch.cuda.is_available():
            print('=============cuda available==================')
            generator = torch.Generator('cuda').manual_seed(42)
        else:
            print('=============cuda not available==============')
            generator = torch.Generator().manual_seed(42)
        print('inference')
        video = self.pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=number_of_frames,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

        file_name = "new_out.mp4"
        export_to_video(video, file_name, fps=fps)

        encoded_frames = encode_video_to_base64(file_name)
        return encoded_frames


