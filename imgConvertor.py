import torch
from PIL import Image
import argparse
import sys

sys.path.append('img2prompt/src/blip')
sys.path.append('img2prompt/clip-interrogator')
from clip_interrogator import Config, Interrogator

class ImgConvertor:
    def __init__(self,device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")):
        self.device = device

        config = Config()
        config.device = self.device
        config.blip_offload = False if torch.cuda.is_available() else True
        config.chunk_size = 2048
        config.flavor_intermediate_count = 512
        config.blip_num_beams = 32
        self.config = config
        self.ci = Interrogator(self.config)
    def inference(self,image_path, mode, best_max_flavors=4):

        ci = self.ci

        image = Image.open(image_path).convert('RGB')

        if mode == 'best':
            prompt_result = ci.interrogate(image, max_flavors=int(best_max_flavors))
        elif mode == 'classic':
            prompt_result = ci.interrogate_classic(image)
        else:  # 'fast'
            prompt_result = ci.interrogate_fast(image)

        # print(f"Mode {mode}: {prompt_result}")
        return prompt_result