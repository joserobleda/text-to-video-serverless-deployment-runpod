# 🧠 Text-to-Video Serverless Deployment (RunPod)

This repository provides a serverless deployment template for **text-to-video generation** using RunPod. It allows you to run a generative video model from a text prompt via a simple API—fully hosted on RunPod's secure, serverless GPU environment.

---

## 🚀 Features

- 🔄 **Text-to-Video** generation using [Model Name — e.g., ModelScope/SVD/AnimateDiff].
- 🌐 **API endpoint** support for easy integration.
- ⚙️ **RunPod serverless template** with `handler.py` and custom Dockerfile.
- 💸 **Low cost & scalable**—only pay when your model is running.

---

## 🛠️ Requirements

- RunPod account: [https://www.runpod.io/](https://www.runpod.io/)
- Docker knowledge (basic)
- Access to a compatible GPU model (depends on video model)
- This repo cloned locally or on GitHub



---

## ⚙️ Deployment Instructions

### 1. Clone this repo

```bash
git clone https://github.com/your-username/text-to-video-runpod-template.git
cd text-to-video-runpod-template
```

### 2. Set up your model (optional)

If needed, modify `download_model.py` to load a specific version from Hugging Face, CivitAI, or another source.

### 3. Create a Serverless Endpoint on RunPod

1. Log in to [RunPod](https://www.runpod.io/).
2. Go to **"Serverless > Community Templates"**.
3. Click **"Create Endpoint"** and select **"Custom Template"**.
4. Upload this repository (or connect your GitHub repo).
5. RunPod will automatically detect `handler.py` and the `Dockerfile`.

### 4. Test the Endpoint

Use the sample `test_input.json` file or send a POST request like this:

```json
 {
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
```

Use `curl` or Postman to test the endpoint.

---

## 📥 Sample API Call

```bash
curl -X POST https://api.runpod.ai/v2/YOUR-ENDPOINT-ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d @test_input.json
```

---

## 📌 Notes

- Depending on the model used, you may need a GPU with at least 24 GB VRAM.
- Video generation time will vary depending on prompt and frame count.
- Make sure your Docker container has `ffmpeg` installed if needed for video encoding.

---


## 📄 License

MIT License. Feel free to modify and use for commercial or personal projects.
