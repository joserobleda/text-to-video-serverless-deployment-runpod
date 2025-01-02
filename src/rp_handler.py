''' infer.py for runpod worker '''

import os
import predict

import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate

from rp_schema import INPUT_SCHEMA


MODEL = predict.Predictor()
MODEL.setup()


def run(job):
    '''
    Run inference on the model.
    Returns output path, width the seed used to generate the image.
    '''
    job_input = job['input']

    # Input validation
    validated_input = validate(job_input, INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    validated_input = validated_input['validated_input']

    # Download input objects
    # job_input['init_image'], job_input['mask'] = rp_download.download_files_from_urls(
    #     job['id'],
    #     [job_input.get('init_image', None), job_input.get('mask', None)]
    # )  # pylint: disable=unbalanced-tuple-unpacking

    # MODEL.NSFW = job_input.get('nsfw', True)

    # if validated_input['seed'] is None:
    #     validated_input['seed'] = int.from_bytes(os.urandom(2), "big")

    encoded_vid = MODEL.predict(
        prompt=validated_input["prompt"],
        num_inference_steps=validated_input['num_inference_steps'],
        number_of_frames=validated_input["number_of_frames"],
        guidance_scale=validated_input['guidance_scale'],
        fps=8
    )

    # job_output = []
    #
    # for index, img_path in enumerate(img_paths):
    #     image_url = rp_upload.upload_image(job['id'], img_path, index)
    #
    #     job_output.append({
    #         "image": image_url,
    #         "seed": validated_input['seed'] + index
    #     })
    #
    # # Remove downloaded input objects
    # rp_cleanup.clean(['input_objects'])

    return encoded_vid


runpod.serverless.start({"handler": run})
