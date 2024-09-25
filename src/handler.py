""" Example handler file. """

import runpod
import torch
import time
from utils import *
from diffusers import CogVideoXPipeline,CogVideoXDDIMScheduler
from diffusers.utils import export_to_video
# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

try:
    print("========Init text to video pipeline======")
    pipe = CogVideoXPipeline.from_pretrained(
        "THUDM/CogVideoX-2b",
        torch_dtype=torch.float16
    )
    pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    pipe.enable_model_cpu_offload()
except RuntimeError:
    quit()


def handler(job):
    """ Handler function that will be used to process jobs. """

    time_start = time.time()
    input_json = job['input']
    updated_json = update_json(default_json, input_json)
    mode = updated_json['mode']
    prompt = updated_json['prompt']
    guidance_scale = updated_json['guidance_scale']
    num_inference_steps = updated_json['num_inference_steps']
    height, width = aspect_ratios[updated_json['aspect_ratio']]
    max_sequence_length = updated_json['max_sequence_length']
    negative_prompt = updated_json["negative_prompt"]
    number_of_frames = updated_json["num_frames"]
    fps = updated_json['fps']
    height, width = aspect_ratios[updated_json['aspect_ratio']]

    height = height - (height % 8)
    width = width - (width % 8)

    # ============== limitation for CogVideoX-2b model =======================
    # fbs should be 8 and max number of frames should be 48
    fps = 8
    number_of_frames = number_of_frames - (number_of_frames % fps)
    number_of_frames = min(number_of_frames, 48)

    try:

        prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=True,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device="cuda",

            dtype=torch.float16,
        )

        video = pipe(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt=negative_prompt,
            num_frames=number_of_frames,
            # height=height,
            # width=width,
        ).frames[0]
        file_name = "new_out.mp4"
        export_to_video(video, file_name, fps=fps)

        print("time elapsed:", time.time() - time_start)
        encoded_frames=encode_video_to_base64(file_name)
        return encoded_frames
    except:

        return {'Comment':'Error Occured'}


runpod.serverless.start({"handler": handler})
