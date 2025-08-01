INPUT_SCHEMA = {
    'prompt': {
        'type': str,
        'required': True
    },
    'negative_prompt': {
        'type': str,
        'required': False,
        'default': None
    },
    'width': {
        'type': int,
        'required': False,
        'default': 768,
        'constraints': lambda width: width in [128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
    },
    'height': {
        'type': int,
        'required': False,
        'default': 768,
        'constraints': lambda height: height in [128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
    },
    'aspect_ratio': {
        'type': str,
        'required': False,
        'default': '16:9',
        'constraints': lambda aspect_ratio: aspect_ratio in ['1:1', '9:16', '2:3', '3:4', '4:3', '3:2', '16:9']
    },
    'prompt_strength': {
        'type': float,
        'required': False,
        'default': 0.8,
        'constraints': lambda prompt_strength: 0 <= prompt_strength <= 1
    },
    'num_outputs': {
        'type': int,
        'required': False,
        'default': 1,
        'constraints': lambda num_outputs: 10 > num_outputs > 0
    },
    'num_inference_steps': {
        'type': int,
        'required': False,
        'default': 32,
        'constraints': lambda num_inference_steps: 0 < num_inference_steps < 500
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 6,
        'constraints': lambda guidance_scale: 0 < guidance_scale < 20
    },
    'fps': {
        'type': int,
        'required': False,
        'default': 8,
        'constraints': lambda fps: 1 <= fps <= 30
    },
    'nsfw': {
        'type': bool,
        'required': False,
        'default': False
    },
    'number_of_frames': {
        'type': int,
        'required': False,
        'default': 49,
        'constraints': lambda number_of_frames: 1 <= number_of_frames <= 200
    }

}
