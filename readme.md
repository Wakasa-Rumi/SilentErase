# SilentErase

# Dataset
visit https://aistudio.baidu.com/datasetdetail/222219 to download WikiArt.zip
images of different artist should be placed under ./dataset

Our trained model for ivan-generalic style is at https://drive.google.com/file/d/1Us_iFINR4V-KpkEx_0-b_6M-nU9Oyo0E/view?usp=drive_link
Please download the model and place it at ./exps_attn1/ivan-generalic/unet/

# Image to Prompt
## Setup
Create and activate a Python virtual environment
```bash
python3 -m venv ci_env
source ci_env/bin/activate
```
Install with PIP
```
cd img2prompt
pip install -e git+https://github.com/pharmapsychotic/BLIP.git@lib#egg=blip
pip install clip-interrogator
```
## Running
```
python img2prompt.py
```
Results will be stored in .\dataset\prompts

# Prompt to Concept
## Setup
1. **Install Python packages:**

```
cd prompt2concept
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
## Running
```
python pro2con.py
```
Results will be stored in .\dataset\concepts

# Attention Resteering
## Set up
```
pip install -r requirements.txt
```
## Running
```
python run.py configs/attn.yaml
```
## Computing Fid
```
python compute_fid.py
```
## Computing Accuracy
```
python compute_concept_score.py
```
