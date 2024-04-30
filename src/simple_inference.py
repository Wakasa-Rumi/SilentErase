import os
import torch
import json
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler

from .patch_lora import safe_open, parse_safeloras_embeds, apply_learned_embed_in_clip
artist_name = 'ivan-generalic'
same_style_name = 'van-gogh'

artists = [
    "van-gogh",
    "claude-monet",
    "antoine-pesne",
    "fernando-botero",
    "ivan-generalic",
    "paul-serusier",
    "pierre-auguste-renoir",
    "thomas-gainsborough",
]

def patch_ti(pipe, ti_paths):
    for weight_path in ti_paths.split('|'):
        token = None
        idempotent_token = True

        safeloras = safe_open(weight_path, framework="pt", device="cpu")
        tok_dict = parse_safeloras_embeds(safeloras)

        apply_learned_embed_in_clip(
            tok_dict,
            pipe.text_encoder,
            pipe.tokenizer,
            token=token,
            idempotent=idempotent_token,
        )

def load_prompt(artist):
    prompt_list = []
    path = "dataset/prompts/{}.json"
    print(path.format(artist))
    with open(path.format(artist), 'rt') as f:
        for line in f:
            prompt_list.append(json.loads(line))
    return prompt_list

def main(args):

    prompts = []

    if args.patch_ti is not None:
        print(f"Inference using Ti {args.pretrained_model_name_or_path}")
        model_id = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
            "cuda"
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)            

        patch_ti(pipe, f"{args.pretrained_model_name_or_path}/step_inv_{args.patch_ti.max_train_steps_ti}.safetensors")

        inverted_tokens = args.patch_ti.placeholder_tokens.replace('|', '')
        if args.patch_ti.use_template == "object":
            prompts += [f"a photo of {inverted_tokens}"]
        elif args.patch_ti.use_template == "style":
            prompts += [f"a photo in the style of {inverted_tokens}"]
        else:
            raise ValueError("unknown concept type!")          

    if args.multi_concept is not None:
        print(f"Inference using {args.pretrained_model_name_or_path}...")
        model_id = args.pretrained_model_name_or_path
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
            "cuda"
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)                       

    prompts = load_prompt(artist_name)[20:]
    print(prompts)

    same_prompts = load_prompt(same_style_name)[20:]

    torch.manual_seed(1)
    output_folder = f"{args.pretrained_model_name_or_path}/generated_images"
    os.makedirs(output_folder, exist_ok=True)

    num = 0
    for prompt_dict in prompts:
        prompt = prompt_dict["prompt"]
        print(f'Inferencing: {prompt}')
        images = pipe(prompt, num_inference_steps=50, guidance_scale=7, num_images_per_prompt=8).images
        for i, im in enumerate(images):
            im.save(f"{output_folder}/{num}.jpg")  
            break
        num += 1

    output_style_folder = f"{args.pretrained_model_name_or_path}/generated_style_images"
    os.makedirs(output_style_folder, exist_ok=True)

    # num = 0
    # for prompt_dict in same_prompts:
    #     prompt = prompt_dict["prompt"]
    #     print(f'Inferencing: {prompt}')
    #     images = pipe(prompt, num_inference_steps=50, guidance_scale=7, num_images_per_prompt=8).images
    #     for i, im in enumerate(images):
    #         im.save(f"{output_style_folder}/{num}.jpg")  
    #         break
    #     num += 1
    #     if num > 3:
    #         break
    
    # for ar in artists:
    #     a_prompts = load_prompt(ar)[20:]
    #     num = 0
    #     cross_style_folder = "/home/yiyao/SOTA/Forget-Me-Not-main/data/cross-set/{}".format(ar)
    #     os.makedirs(cross_style_folder, exist_ok=True)
    #     for prompt_dict in a_prompts:
    #         prompt = prompt_dict["prompt"]
    #         print(f'Inferencing: {prompt}')
    #         images = pipe(prompt, num_inference_steps=50, guidance_scale=7, num_images_per_prompt=8).images
    #         for i, im in enumerate(images):
    #             im.save("/home/yiyao/SOTA/Forget-Me-Not-main/data/cross-set/{}/{}.jpg".format(ar, num))  
    #             break
    #         num += 1