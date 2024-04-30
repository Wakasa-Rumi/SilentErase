import torch
from PIL import Image
import argparse
import sys
import os
import json

import shutil

sys.path.append('img2prompt/src/blip')
sys.path.append('img2prompt/clip-interrogator')
from clip_interrogator import Config, Interrogator

def model_init():
    config = Config()
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.blip_offload = False if torch.cuda.is_available() else True
    config.chunk_size = 2048
    config.flavor_intermediate_count = 512
    config.blip_num_beams = 32
    ci = Interrogator(config)
    return ci

def inference(ci, image_path, mode, best_max_flavors=4):
    prompt_list = []
    image = Image.open(image_path).convert('RGB')
    
    if mode == 'best':
        prompt_result = ci.interrogate(image, max_flavors=int(best_max_flavors))
    elif mode == 'classic':
        prompt_result = ci.interrogate_classic(image)
    else:  # 'fast'
        prompt_result = ci.interrogate_fast(image)
    
    prompt_json = {}
    prompt_json["prompt"] = prompt_result
    prompt_list.append(prompt_json)
    print(f"Mode {mode}: {prompt_result}")
    return prompt_list

def dump(artist, prompt_list):
    with open("data/cross-set/{}.json".format(artist),"a+", newline='\n') as f1:
        for i in range(0, len(prompt_list)):
            json_str = json.dumps(prompt_list[i])
            f1.write("\n" + json_str)

def main():
    parser = argparse.ArgumentParser(description="Run inference with CLIP Interrogator")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("--mode", type=str, choices=['best', 'classic', 'fast'], default='fast', help="Inference mode")
    parser.add_argument("--best_max_flavors", type=int, default=4, help="Maximum flavors for 'best' mode")
    args = parser.parse_args()

    inference(args.image_path, args.mode, args.best_max_flavors)

def traverse_folder(folder_path):
    train_dir_list = []
    num = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            train_dir_list.append(file_path)
            # if num > 19:
            #     shutil.copy(file_path,'data/test_set/{}/{}.jpg'.format(artist, num-20))
            # else:
            #     shutil.copy(file_path,'data/train_set/{}/{}.jpg'.format(artist, num))
            num += 1
    # print(train_dir_list)
    return train_dir_list

def get_prompt(ci, artist):
    train_dir_list = traverse_folder('data/cross-set/{}/'.format(artist))
    for path in train_dir_list:
        prompt_list = inference(ci, path, 'best', 4)
        dump(artist, prompt_list)

artists = [
    "antoine-pesne",
    "claude-monet",
    "fernando-botero",
    "ivan-generalic",
    "paul-serusier",
    "pierre-auguste-renoir",
    "thomas-gainsborough",
    "van-gogh"
]
def split_data():
    for artist in artists:
        print(artist)
        path = "dataset/{}".format(artist)
        traverse_folder(artist, path)

if __name__ == "__main__":
    ci = model_init()
    get_prompt(ci)
    traverse_folder('dataset/claude-monet')
    # split_data()
    # for artist in artists:
    #     print(artist)
    #     get_prompt(ci, artist)