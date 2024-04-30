import torch
import json
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def init():
    model_id = "stabilityai/stable-diffusion-2-1-base"

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    return pipe

prompt_list = []
def load_prompt():
    with open('/home/yiyao/SOTA/Forget-Me-Not-main/dataset/prompts/antoine-pesne.json', 'rt') as f:
        for line in f:
            prompt_list.append(json.loads(line))

def generate_fake_images(pipe):
    load_prompt()

    for i in range(0, len(prompt_list)):
        prompt = prompt_list[i]["prompt"]
        image = pipe(prompt).images[0]
        image.save("/home/yiyao/SOTA/Forget-Me-Not-main/dataset_fake/antoine-pesne/{}.jpg".format(i))

def image_instance(pipe):
    prompt = "a photo in the style of Paul Gauguin."
    image = pipe(prompt).images[0]
        
    image.save("ghibli.png")

if __name__ == '__main__':
    pipe = init()
    generate_fake_images(pipe)
    # image_instance(pipe)

# from transformers import CLIPTextModel

# model_id = "stabilityai/stable-diffusion-2-1-base"
# model = CLIPTextModel
# encoder = model.from_pretrained(model_id, subfolder="image_encoder", revision=None)