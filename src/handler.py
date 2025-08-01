""" Example handler file. """

import runpod
import time
from utils import *
import torch
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

# try:
print("========Init text to video pipeline======")
if torch.cuda.is_available():
    print('=============cuda available==================')
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)
# except RuntimeError:
#     quit()
print("========loaded======")


def handler(job):
    """ Handler function that will be used to process jobs. """
    print("========Job starting======")

    time_start = time.time()
    input_json = job['input']
    
    # Extract parameters from input
    prompt = input_json.get('prompt', None)
    guidance_scale = input_json.get('guidance_scale', 6)
    num_inference_steps = input_json.get('num_inference_steps', 32)
    number_of_frames = input_json.get('num_frames', 49)
    aspect_ratio = input_json.get('aspect_ratio', '16:9')  # Default to 16:9
    fps = input_json.get('fps', 8)  # Default to 8
    
    # Get height and width from aspect ratio
    if aspect_ratio in aspect_ratios:
        height, width = aspect_ratios[aspect_ratio]
    else:
        # Default to 16:9 if aspect ratio not found
        print(f"Aspect ratio {aspect_ratio} not found, using default 16:9")
        height, width = aspect_ratios['16:9']
    
    # Ensure height and width are divisible by 8 (requirement for the model)
    height = height - (height % 8)
    width = width - (width % 8)

    # ============== limitation for CogVideoX-5b model =======================
    # Validate fps (keeping the original constraint but allowing customization)
    if fps <= 0:
        fps = 8
        print("Invalid fps provided, using default fps=8")
    
    # Validate number of frames
    if number_of_frames <= 0:
        number_of_frames = 49
        print("Invalid num_frames provided, using default num_frames=49")

    print(f"Using parameters: aspect_ratio={aspect_ratio}, height={height}, width={width}, fps={fps}, frames={number_of_frames}")

    # try:
    if torch.cuda.is_available():
        print('=============cuda available==================')
        generator = torch.Generator('cuda').manual_seed(42)
    else:
        print('=============cuda not available==============')
        generator = torch.Generator().manual_seed(42)
    print('inference')
    video = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=number_of_frames,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width,
    ).frames[0]

    file_name = "new_out.mp4"
    export_to_video(video, file_name, fps=fps)

    print("time elapsed:", time.time() - time_start)
    encoded_frames=encode_video_to_base64(file_name)
    return encoded_frames
    # except:
    #
    #     return {'Comment':'Error Occured'}


runpod.serverless.start({"handler": handler})
