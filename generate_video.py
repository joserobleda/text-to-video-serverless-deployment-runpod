

# for local testing
import asyncio
import aiohttp
import os
import runpod
from runpod import AsyncioEndpoint, AsyncioJob
import time

# add your api key and endpoint
runpod.api_key = ""
end_point=''
import base64
from PIL import Image
from io import BytesIO


def decode_base64_to_video(base64_string, output_file_path):
    """
    Decodes a Base64 string to a video file.
    :param base64_string: Base64 encoded string.
    :param output_file_path: Path to save the decoded video file.
    """
    # Decode Base64 string to binary data
    video_bytes = base64.b64decode(base64_string)

    # Write the binary data to a video file
    with open(output_file_path, 'wb') as video_file:
        video_file.write(video_bytes)

async def main():
    start=time.time()
    async with aiohttp.ClientSession() as session:

        input_payload = {
            "mode": "txt2video",
            "prompt": "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance",
            "negative_prompt": "",
            "num_frames": 48,
            "guidance_scale": 6,
            "aspect_ratio": "1:1",
            "num_inference_steps": 50,
            "max_sequence_length": 226,
            "fps": 8

        }
        endpoint = AsyncioEndpoint(end_point, session)
        job: AsyncioJob = await endpoint.run(input_payload)

        # Polling job status
        while True:
            status = await job.status()
            print(f"Current job status: {status}")
            if status == "COMPLETED":
                output = await job.output()
                print("Job output:", output)
                decode_base64_to_video(output,'output.mp4')
                break  # Exit the loop once the job is completed.
            elif status in ["FAILED"]:
                print("Job failed or encountered an error.")

                break
            else:
                print("Job in queue or processing. Waiting 3 seconds...")
                await asyncio.sleep(3)  # Wait for 3 seconds before polling again
    print("time elapsed:",time.time()-start)

if __name__ == "__main__":
    asyncio.run(main())
