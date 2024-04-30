# Ner Utils
Extract the style word from prompt

## Setup
1. **Install Python packages:**

```
pip install -r requirements.txt
```
2. **Apply for the access to llama and login**
```
    1.Visit https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
    2.Apply for the access to the model(it may take 1-2 hours)
    2.Use huggingface cli to login
```
2. **Creating the object of the util class**
```
from prompt2concept.ner_utils import StyleExtractor
device1 = torch.device("cuda")
styleExtractor = StyleExtractor(device = device)
```
3. **Extract style words from prompts**
```
raw_prompts = "Mode fast: a couple of anime characters standing next to each other, howl’s moving castle, howl's moving castle, howls moving castle, moving castle, style in ghibli anime, style in ghibli anime style, ghibli animated film, miyazaki film, miyazaki's animated film, ghibli moebius, ghibli studio anime style, ghibli film, by Miyazaki"
styleExtractor.prompt2concepts(raw_prompts)
# output:
{
'concepts': ['edge anime','ghibli anime','miyazaki film','moebius','ghibli studio'],
'concepts_index': [(158, 169), (223, 235), (271, 277), (280, 292)]
}

```
