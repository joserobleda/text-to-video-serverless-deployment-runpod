''' infer.py for runpod worker '''

import os
import io
import base64
import predict

import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_in_memory_object

# direct S3 upload
import boto3
from botocore.exceptions import ClientError

from rp_schema import INPUT_SCHEMA


def upload_video(video_file_path, key):
    """ Uploads video to Cloudflare R2 bucket if it is available, otherwise returns base64 encoded video. """
    
    # Read video file as bytes
    with open(video_file_path, 'rb') as video_file:
        video_bytes = video_file.read()

    # Upload to Cloudflare R2 (S3-compatible) - Direct upload to avoid date folders
    if os.environ.get('BUCKET_ENDPOINT_URL', False):
        try:
            # Parse endpoint and bucket from URL
            endpoint_url = os.environ.get('BUCKET_ENDPOINT_URL')
            
            # If the URL contains a bucket name, extract it
            if '/' in endpoint_url.split('://', 1)[1]:
                # URL format: https://account-id.r2.cloudflarestorage.com/bucket-name
                base_url, bucket_name = endpoint_url.rsplit('/', 1)
                actual_endpoint = base_url
            else:
                # URL format: https://account-id.r2.cloudflarestorage.com
                actual_endpoint = endpoint_url
                bucket_name = os.environ.get('BUCKET_NAME', 'tts')
            
            # Create S3 client for direct upload
            s3_client = boto3.client(
                's3',
                endpoint_url=actual_endpoint,
                aws_access_key_id=os.environ.get('BUCKET_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('BUCKET_SECRET_ACCESS_KEY'),
                region_name='auto'  # Cloudflare R2 uses 'auto' as region
            )
            
            # Upload directly to root of bucket (no date folders)
            s3_client.put_object(
                Bucket=bucket_name,
                Key=key,  # File will go directly to root with this key
                Body=video_bytes,
                ContentType='video/mp4'
            )
            
            # Return the public URL
            return f"{actual_endpoint}/{bucket_name}/{key}"
            
        except Exception as e:
            print(f"Direct S3 upload failed: {e}")
            print("Falling back to RunPod upload function...")
            # Fallback to original RunPod function
            return upload_in_memory_object(
                key,
                video_bytes,
                bucket_creds = {
                    "endpointUrl": os.environ.get('BUCKET_ENDPOINT_URL', None),
                    "accessId": os.environ.get('BUCKET_ACCESS_KEY_ID', None),
                    "accessSecret": os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)
                }
            )
    
    # Base64 encode for direct return if no bucket configured
    return base64.b64encode(video_bytes).decode('utf-8')


MODEL = predict.Predictor()
MODEL.setup()


def run(job):
    '''
    Run inference on the model.
    Returns video URL after uploading to R2 bucket.
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

    # Run prediction (this creates the video file and returns base64)
    encoded_vid = MODEL.predict(
        prompt=validated_input["prompt"],
        num_inference_steps=validated_input['num_inference_steps'],
        number_of_frames=validated_input["number_of_frames"],
        guidance_scale=validated_input['guidance_scale'],
        fps=8
    )

    # Upload the video file to R2 bucket
    video_file_path = "new_out.mp4"  # This is the file created by predict()
    video_key = f"video_{job['id']}.mp4"  # Use job ID for unique filename
    
    try:
        video_url = upload_video(video_file_path, video_key)
        
        # Clean up local video file
        if os.path.exists(video_file_path):
            os.remove(video_file_path)
        
        return {"video_url": video_url}
        
    except Exception as e:
        print(f"Video upload failed: {e}")
        # Fallback to base64 if upload fails
        return {"video_base64": encoded_vid}

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


runpod.serverless.start({"handler": run})
