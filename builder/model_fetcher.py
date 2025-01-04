'''
RunPod | serverless-ckpt-template | model_fetcher.py

Downloads the model from the URL passed in.
'''


from huggingface_hub import snapshot_download



if __name__ == "__main__":
    model_path = 'model_cache'  # The local directory to save downloaded checkpoint
    snapshot_download("THUDM/CogVideoX-5b", local_dir=model_path, local_dir_use_symlinks=False, repo_type='model')
