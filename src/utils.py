import json
import base64
from io import BytesIO


# Step 1: Define default JSON template
default_json = {
    "mode":"txt2video",
    "prompt":"A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance",
    "negative_prompt":"",
    "num_frames":48,
    "guidance_scale":6,
    "aspect_ratio":"1:1",
    "num_inference_steps":50,
    "max_sequence_length":226,
    "fps":8

}

aspect_ratios={

               "1:1":[1024,1024],
               "9:16":[720,1280],
               "2:3":[836,1254],
               "3:4":[876,1168],
               "4:3":[1168,876],
               "3:2":[1254,836],
               "16:9":[1280,720]

}

# Step 2: Function to update JSON based on another JSON
def update_json(base_json, update_json):
    for key, value in update_json.items():
        if key in base_json:
            base_json[key] = value
    return base_json




def encode_video_to_base64(video_file_path):
    """
    Encodes a video file to a Base64 string.
    :param video_file_path: Path to the video file.
    :return: Base64 encoded string.
    """
    with open(video_file_path, 'rb') as video_file:
        # Read the video file as binary
        video_bytes = video_file.read()

        # Encode binary data to Base64
        base64_encoded = base64.b64encode(video_bytes).decode('utf-8')

    return base64_encoded