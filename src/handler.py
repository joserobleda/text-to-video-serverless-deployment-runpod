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
    # updated_json = update_json(default_json, input_json)
    # mode = updated_json['mode']
    prompt = input_json.get('prompt',None)
    guidance_scale = input_json.get('guidance_scale',6)
    num_inference_steps =input_json.get('num_inference_steps',32)
    # height, width = aspect_ratios[updated_json['aspect_ratio']]
    # max_sequence_length = updated_json['max_sequence_length']
    # negative_prompt = updated_json["negative_prompt"]
    number_of_frames = input_json.get('num_frames',49)
    # fps = updated_json['fps']
    # height, width = aspect_ratios[updated_json['aspect_ratio']]
    #
    # height = height - (height % 8)
    # width = width - (width % 8)

    # ============== limitation for CogVideoX-2b model =======================
    # fbs should be 8 and max number of frames should be 48
    fps = 8
    # number_of_frames = number_of_frames - (number_of_frames % fps)
    # number_of_frames = min(number_of_frames, 48)

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
