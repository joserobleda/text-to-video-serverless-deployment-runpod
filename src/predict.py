import os
from typing import List
from utils import encode_video_to_base64
import torch
import numpy as np
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video, load_image
from transformers import T5EncoderModel, T5Tokenizer
import gc  # Add garbage collection import
model_path = 'model_cache'   # The local directory to save downloaded checkpoint

model_id=model_path

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
        transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer",
                                                                  torch_dtype=torch.float16)
        text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)
        vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)
        tokenizer = T5Tokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.pipe = CogVideoXPipeline.from_pretrained(model_id, tokenizer=tokenizer, text_encoder=text_encoder,
                                                 transformer=transformer, vae=vae, torch_dtype=torch.float16).to("cuda")


        # self.pipe.enable_xformers_memory_efficient_attention()

    @torch.inference_mode()
    def predict(self, prompt, number_of_frames, num_inference_steps, guidance_scale, fps, negative_prompt=None):
        if torch.cuda.is_available():
            print('=============cuda available==================')
            generator = torch.Generator('cuda').manual_seed(42)
        else:
            print('=============cuda not available==============')
            generator = torch.Generator().manual_seed(42)
        
        # Progress callback function (officially supported by CogVideoX)
        def progress_callback(pipeline, step, timestep, callback_kwargs):
            progress_percent = ((step + 1) / num_inference_steps) * 100
            print(f"🎬 Generation Progress: {progress_percent:.1f}% ({step + 1}/{num_inference_steps} steps)")
            return callback_kwargs
        
        print(f'🚀 Starting inference: {num_inference_steps} steps, {number_of_frames} frames')
        
        # Prepare pipeline arguments
        pipe_args = {
            "prompt": prompt,
            "num_videos_per_prompt": 1,
            "num_inference_steps": num_inference_steps,
            "num_frames": number_of_frames,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "callback_on_step_end": progress_callback,
            "callback_on_step_end_tensor_inputs": ["latents"],  # Required for callback
        }
        
        # Only add negative_prompt if it's not None/empty
        if negative_prompt and negative_prompt.strip():
            pipe_args["negative_prompt"] = negative_prompt
        
        video = self.pipe(**pipe_args).frames[0]
        print("✅ Generation completed! Exporting video...")

        file_name = "new_out.mp4"
        export_to_video(video, file_name, fps=fps)
        print(f"🎥 Video exported at {fps}fps: {file_name}")

        encoded_frames = encode_video_to_base64(file_name)
        
        # ===== MEMORY CLEANUP =====
        # Clear GPU cache and free memory to prevent accumulation between jobs
        try:
            print("Starting GPU memory cleanup...")
            
            # Clear large variables first
            if 'video' in locals():
                del video
            if 'generator' in locals():
                del generator
            
            # PyTorch GPU memory cleanup
            if torch.cuda.is_available():
                # Clear PyTorch GPU cache
                torch.cuda.empty_cache()
                # Synchronize to ensure cleanup is complete
                torch.cuda.synchronize()
                
                # Get memory info for debugging
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_cached = torch.cuda.memory_reserved() / 1024**3      # GB
                print(f"GPU Memory after cleanup - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
            
            # Force garbage collection
            gc.collect()
            
            print("GPU memory cleanup completed")
            
        except Exception as cleanup_error:
            print(f"Warning: Memory cleanup failed: {cleanup_error}")
        # ===== END MEMORY CLEANUP =====
        
        print(f"🎬 VIDEO GENERATION COMPLETE! ({number_of_frames} frames, {fps}fps)")
        return encoded_frames


