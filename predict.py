# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, BaseModel, Input, Path
from typing import Optional, List
import os
import PanoHead.dnnlib as dnnlib
import torch

import sys

class ModelOutput(BaseModel):
    training_video: Path
    preview_video: Path
    ply_out: Optional[Path]

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # Download VGG16 feature detector.
        os.chdir("3DDFA_V2")
        os.system("sh ./build.sh")
        os.chdir("..")
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval()
        del vgg16


    def predict(
        self,
        face_image: Path = Input(description="Input subject face image"),
        create_ply: bool = Input(
            description="If you want to save the 3D mesh as PLY", default=False
        ),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        os.system("rm -rf stage/* 3DDFA_V2/crop_samples/img/* 3DDFA_V2/test/original/* output/*")

        print("Cropping...")
        os.chdir("/src/3DDFA_V2")
        sys.path.append(os.getcwd())

        os.system(f"cp {face_image} test/original")
        os.system("python dlib_kps.py")
        os.system("python recrop_images.py -i data.pkl -j dataset.json")

        os.chdir("..")
        os.system("cp -rf 3DDFA_V2/crop_samples/img/* stage")
        os.chdir("PanoHead")

        if create_ply:
            add = "--shapes=True "
        else:
            add=""
        
        # Tuning + video
        print("Projection...")
        os.system(f"python projector_withseg.py {add}--num-steps=300 --num-steps-pti=300 --outdir=/src/output --target_img=/src/stage --network=/src/PanoHead/models/easy-khair-180-gpc0.8-trans10-025000.pkl --idx 0")
        
        # 360 video
        print("Create video...")
        os.system("python gen_videos_proj_withseg.py --output=/src/output/post.mp4 --latent=/src/output/easy-khair-180-gpc0.8-trans10-025000.pkl/0/projected_w.npz --trunc 0.7 --network /src/output/easy-khair-180-gpc0.8-trans10-025000.pkl/0/fintuned_generator.pkl --cfg Head")
        
        if create_ply:
            ply = Path("/src/output/easy-khair-180-gpc0.8-trans10-025000.pkl/0/projected_w.npz")
        else:
            ply = None
        
        return ModelOutput(training_video=Path("/src/output/easy-khair-180-gpc0.8-trans10-025000.pkl/0/proj.mp4"), preview_video=Path("/src/output/post.mp4"), ply_out=ply)
