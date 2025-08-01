''' infer.py for runpod worker '''

import os
import io
import base64
import predict
import gc  # Add garbage collection import

import runpod
from runpod.serverless.utils import rp_download, rp_upload, rp_cleanup
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.utils.rp_upload import upload_in_memory_object

# direct S3 upload
import boto3
from botocore.exceptions import ClientError
import torch  # Add torch import for GPU memory cleanup

from rp_schema import INPUT_SCHEMA


def upload_video(video_file_path, key):
    """ Uploads video to Cloudflare R2 bucket if it is available, otherwise returns base64 encoded video. """
    
    print(f"upload_video called with file: {video_file_path}, key: {key}")
    
    # Read video file as bytes
    with open(video_file_path, 'rb') as video_file:
        video_bytes = video_file.read()
    
    print(f"Read video file, size: {len(video_bytes)} bytes")

    # Upload to Cloudflare R2 (S3-compatible) - Direct upload to avoid date folders
    bucket_endpoint = os.environ.get('BUCKET_ENDPOINT_URL', False)
    print(f"BUCKET_ENDPOINT_URL: {bucket_endpoint}")
    
    if bucket_endpoint:
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
            
            print(f"Using endpoint: {actual_endpoint}, bucket: {bucket_name}")
            
            # Create S3 client for direct upload
            s3_client = boto3.client(
                's3',
                endpoint_url=actual_endpoint,
                aws_access_key_id=os.environ.get('BUCKET_ACCESS_KEY_ID'),
                aws_secret_access_key=os.environ.get('BUCKET_SECRET_ACCESS_KEY'),
                region_name='auto'  # Cloudflare R2 uses 'auto' as region
            )
            
            print(f"S3 client created, uploading to bucket '{bucket_name}' with key '{key}'")
            
            # Upload directly to root of bucket (no date folders)
            s3_client.put_object(
                Bucket=bucket_name,
                Key=key,  # File will go directly to root with this key
                Body=video_bytes,
                ContentType='video/mp4'
            )
            
            # Return the public URL
            video_url = f"{actual_endpoint}/{bucket_name}/{key}"
            print(f"S3 upload successful, returning URL: {video_url}")
            return video_url
            
        except Exception as e:
            print(f"Direct S3 upload failed: {e}")
            print("Falling back to RunPod upload function...")
            # Fallback to original RunPod function
            try:
                result = upload_in_memory_object(
                    key,
                    video_bytes,
                    bucket_creds = {
                        "endpointUrl": os.environ.get('BUCKET_ENDPOINT_URL', None),
                        "accessId": os.environ.get('BUCKET_ACCESS_KEY_ID', None),
                        "accessSecret": os.environ.get('BUCKET_SECRET_ACCESS_KEY', None)
                    }
                )
                print(f"RunPod upload successful: {result}")
                return result
            except Exception as e2:
                print(f"RunPod upload also failed: {e2}")
                raise e2
    else:
        print("No BUCKET_ENDPOINT_URL configured, returning base64")
    
    # Base64 encode for direct return if no bucket configured
    encoded = base64.b64encode(video_bytes).decode('utf-8')
    print(f"Returning base64 encoded video (length: {len(encoded)})")
    return encoded


MODEL = predict.Predictor()
MODEL.setup()


def run(job):
    '''
    Run inference on the model.
    Returns video URL after uploading to R2 bucket.
    '''
    print(f"=== RUN FUNCTION CALLED ===")
    print(f"Job ID: {job.get('id', 'unknown')}")
    print(f"Job input: {job.get('input', {})}")
    
    try:
        job_input = job['input']
        print(f"Extracted job_input: {job_input}")

        # Input validation
        print("Starting input validation...")
        validated_input = validate(job_input, INPUT_SCHEMA)
        print(f"Validation result: {validated_input}")

        if 'errors' in validated_input:
            print(f"Validation errors found: {validated_input['errors']}")
            return {"error": validated_input['errors']}
        validated_input = validated_input['validated_input']
        print(f"Validated input: {validated_input}")

        # Download input objects
        # job_input['init_image'], job_input['mask'] = rp_download.download_files_from_urls(
        #     job['id'],
        #     [job_input.get('init_image', None), job_input.get('mask', None)]
        # )  # pylint: disable=unbalanced-tuple-unpacking

        # MODEL.NSFW = job_input.get('nsfw', True)

        # if validated_input['seed'] is None:
        #     validated_input['seed'] = int.from_bytes(os.urandom(2), "big")

        # Run prediction (this creates the video file and returns base64)
        print("Starting MODEL.predict...")
        encoded_vid = MODEL.predict(
            prompt=validated_input["prompt"],
            num_inference_steps=validated_input['num_inference_steps'],
            number_of_frames=validated_input["number_of_frames"],
            guidance_scale=validated_input['guidance_scale'],
            fps=validated_input.get("fps", 8)
        )
        print("MODEL.predict completed")

        # Upload the video file to R2 bucket
        video_file_path = "new_out.mp4"  # This is the file created by predict()
        video_key = f"video_{job['id']}.mp4"  # Use job ID for unique filename
        
        print(f"Attempting to upload video file: {video_file_path}")
        print(f"Video file exists: {os.path.exists(video_file_path)}")
        
        if os.path.exists(video_file_path):
            file_size = os.path.getsize(video_file_path)
            print(f"Video file size: {file_size} bytes")
        
        try:
            print(f"Uploading with key: {video_key}")
            video_url = upload_video(video_file_path, video_key)
            print(f"Upload successful, video URL: {video_url}")
            
            # Clean up local video file
            if os.path.exists(video_file_path):
                os.remove(video_file_path)
                print("Local video file cleaned up")
            
            result = {"video_url": video_url}
            print(f"Returning result: {result}")
            
            # ===== ENHANCED MEMORY CLEANUP =====
            # Force memory cleanup to prevent accumulation between jobs
            try:
                print("Starting post-job memory cleanup...")
                
                # Clear any remaining variables
                if 'encoded_vid' in locals():
                    del encoded_vid
                if 'video_url' in locals() and 'result' in locals():
                    # Keep result but clear other references
                    pass
                
                # PyTorch GPU memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Wait for GPU operations to complete
                    
                    # Get memory stats for monitoring
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_cached = torch.cuda.memory_reserved() / 1024**3      # GB
                    print(f"Post-job GPU Memory - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
                
                # Force garbage collection
                gc.collect()
                
                print(f"Memory cleanup completed for job {job['id']}")
                
            except Exception as cleanup_error:
                print(f"Warning: Post-job memory cleanup failed: {cleanup_error}")
            # ===== END ENHANCED MEMORY CLEANUP =====
            
            return result
            
        except Exception as upload_e:
            print(f"Video upload failed: {upload_e}")
            print(f"Falling back to base64 encoded video")
            # Fallback to base64 if upload fails
            result = {"video_base64": encoded_vid}
            print(f"Returning fallback result with base64 (length: {len(encoded_vid) if encoded_vid else 0})")
            
            # ===== ENHANCED MEMORY CLEANUP (FALLBACK PATH) =====
            try:
                print("Starting post-job memory cleanup (fallback path)...")
                
                # Clear variables
                if 'encoded_vid' in locals() and 'result' in locals():
                    # Keep result but clear other references
                    pass
                
                # PyTorch GPU memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_cached = torch.cuda.memory_reserved() / 1024**3
                    print(f"Post-job GPU Memory (fallback) - Allocated: {memory_allocated:.2f}GB, Cached: {memory_cached:.2f}GB")
                
                gc.collect()
                print(f"Memory cleanup completed for job {job['id']} (fallback path)")
                
            except Exception as cleanup_error:
                print(f"Warning: Post-job memory cleanup failed (fallback): {cleanup_error}")
            # ===== END ENHANCED MEMORY CLEANUP (FALLBACK PATH) =====
            
            return result

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
    
    except Exception as e:
        print(f"CRITICAL ERROR in run function: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        
        # ===== EMERGENCY MEMORY CLEANUP =====
        try:
            print("Starting emergency memory cleanup after error...")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            print("Emergency memory cleanup completed")
        except Exception as emergency_cleanup_error:
            print(f"Warning: Emergency memory cleanup failed: {emergency_cleanup_error}")
        # ===== END EMERGENCY MEMORY CLEANUP =====
        
        return {"error": f"Handler crashed: {str(e)}"}


runpod.serverless.start({"handler": run})
