from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="skin_disease_model_final.pth",  # Local file path
    path_in_repo="skin_disease_model_final.pth",     # Where it will sit inside the repo
    repo_id="samlowkey/skin-disease",
    repo_type="model"
)