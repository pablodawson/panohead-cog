import os
from huggingface_hub import hf_hub_download
import joblib

# Repos
os.system("git clone -b dev1 https://github.com/camenduru/PanoHead")
os.system("git clone -b dev https://github.com/camenduru/3DDFA_V2")

# Move resources to 3DDFA
os.system("cp -rf PanoHead/3DDFA_V2_cropping/test 3DDFA_V2")
os.system("cp PanoHead/3DDFA_V2_cropping/dlib_kps.py 3DDFA_V2")
os.system("cp PanoHead/3DDFA_V2_cropping/recrop_images.py 3DDFA_V2")

# Download models
models_dir = "PanoHead/models"
REPO_ID = "camenduru/PanoHead"
MODELS = ["ablation-trigridD-1-025000.pkl", "baseline-easy-khair-025000.pkl", "easy-khair-180-gpc0.8-trans10-025000.pkl"]

for model in MODELS:
    dl = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=model, local_dir=models_dir)
    )

# Face landmarks
REPO_ID = "camenduru/shape_predictor_68_face_landmarks"

dl = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename="shape_predictor_68_face_landmarks.dat", local_dir="3DDFA_V2")
)


# Make in/out folders
os.system("mkdir in stage output")