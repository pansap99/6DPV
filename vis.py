import tyro
import dataclasses
import glob
import json
import numpy as np
from utils.vis_modes import read_models_vertices, build_cuboid_from_model, draw_cuboid3D_on_image
import cv2

@dataclasses.dataclass
class vis_args:
    
    # Visualization mode. Can be a combination of two or more.
    mode: str = 'cuboid'
    
    # Model file
    model_file: str = './model.ply'
    
    # image
    image_file: str = './image.png'
    
  
def run(args):
    
    model = read_models_vertices(args.model_file)
    cuboid = build_cuboid_from_model(model,axis_up='z')
    pose = np.array([0.39304935932159424, -0.6464345455169678, -0.6539376974105835, 75.31982421875,
            -0.8118026256561279, -0.5779479742050171, 0.08338263630867004, 24.96088409423828,
            -0.4318433701992035, 0.4980948865413666, -0.7519392967224121, 740.013916015625,
            0,0,0,1]).reshape(4,4)
    print(pose)
    Kmat = np.array([422.74438753, 0.0, 318.68035889, 0.0, 422.42329915, 242.97497559, 0.0, 0.0, 1]).reshape(3,3)
    draw_cuboid3D_on_image(args.image_file, cuboid, pose, Kmat, thickness=1, color=(255,0,0))
        

if __name__ == "__main__":
    args = tyro.cli(vis_args)
    run(args)