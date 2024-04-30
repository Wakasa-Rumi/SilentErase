import os
import torch
import json
# os.chdir("forget-from-image")
from prompt2concept.ner_utils import StyleExtractor

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

path = "dataset/prompts/{}.json"
# artist = "antoine-pesne"

def load_prompt(artist):
    prompt_list = []
    print(path.format(artist))
    with open(path.format(artist), 'rt') as f:
        for line in f:
            prompt_list.append(json.loads(line))
    return prompt_list

if __name__ == '__main__':
    device = torch.device("cuda:0")
    styleExtractor = StyleExtractor(device = device)

    for artist in artists:
        concept_dict = {}
        prompt_list = load_prompt(artist)

        for prompt in prompt_list:
            raw_prompts = prompt["prompt"]
            concepts = styleExtractor.prompt2concepts(raw_prompts)
            # print(concepts)
            if concepts is not None and concepts['concepts'] is not None:
                for concept in concepts['concepts']: 
                    if concept in concept_dict.keys():
                        concept_dict[concept] += 1
                    else:
                        concept_dict[concept] = 1
        
        # print(concept_dict)
        with open("dataset/concepts/{}.json".format(artist),"w") as f:
            json.dump(concept_dict, f, indent=4, ensure_ascii=False)