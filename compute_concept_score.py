import json

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
styles = {
    "antoine-pesne": ["rococo", "antoine", "pesne"],
    "claude-monet": ["Impressionism", "claude monet", "claude", "monet"],
    "fernando-botero": ["naive art primitivism", "naive", "fernando botero", "fernando", "botero"],
    "ivan-generalic": ["naive art primitivism", "naive", "ivan", "generalic"],
    "paul-serusier": ["post Impressionism", "post-Impressionism", "paul serusier", "paul", "serusier"],
    "pierre-auguste-renoir": ["Impressionism", "pierre auguste renoir", "pierre", "auguste", "renoir"],
    "thomas-gainsborough": ["rococo", "thomas gainsborough", "thomas", "gainsborough"],
    "van-gogh": ["post Impressionism", "post-Impressionism", "van gogh", "van", "gogh"]
}
path = "/home/yiyao/SOTA/Forget-Me-Not-main/data/cross-set/{}.json"

def detect(artist, prompt):
    style = styles[artist]
    p = prompt.lower()
    for s in style:
        if p.find(s) != -1:
            return 1
    return 0

def load_prompt(artist):
    prompt_list = []
    print(path.format(artist))
    with open(path.format(artist), 'rt') as f:
        for line in f:
            prompt_list.append(json.loads(line))
    return prompt_list

if __name__ == '__main__':
    for ar in artists:
        num = 0
        prompt_list = load_prompt(ar)
        for pd in prompt_list:
            prompt = pd["prompt"]
            num += detect(ar, prompt)
        print("ar: ", num / 20)