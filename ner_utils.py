import torch
from transformers import pipeline
import re

class StyleExtractor:
    def __init__(self,model_name="meta-llama/Llama-2-7b-chat-hf",device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.model= pipeline("text-generation", model=model_name, device=device, torch_dtype=torch.float16)
    @staticmethod
    def prompt_builder(oneshot,user_message):
        prompt_template = '''
        <<SYS>>You are a powerful named entity recognizer,you are to extract the words that describe the painting style,do output only the answer without explain.You should output at most five answers in python list.{oneshot}<</SYS>> 
        [INST] User:{user_message} [/INST]\n Answer:
        '''
        return prompt_template.format(oneshot=oneshot,user_message=user_message)
    @staticmethod
    def oneshot_builder(question,answer):
        return f"Question:{question},Answer:{answer}"
    @staticmethod
    def extract_answers(text):
        answer_index = text.find("Answer:\n")
        answer_text = text[answer_index:]
        regrex_pattern = r"\[[a-z ,A-Z]*\]"
        answer = re.findall(regrex_pattern, answer_text)[0]
        answer = answer.lower()
        answer=answer.replace("[","")
        answer=answer.replace("]","")
        split_answer = answer.split(",")
        split_answer = list(map(str.strip, split_answer))
        return split_answer
    @staticmethod
    def answer2index(answers,raw_prompt):
        result =[]
        for answer in answers:
            start_index = raw_prompt.find(answer)
            if start_index == -1:
                continue
            result.append((start_index,len(answer)+start_index-1))
        return result
    def prompt2concepts(self,raw_prompt):
        oneshot = self.oneshot_builder(
            "woman with blue hair, in the style of multi-layered collages, edgy street art, celebrity-portraits, cardboard, fragmented icons, realistic hyper-detail, crossed colors --ar 73:111 --stylize 750 --v 6",
            "[edge street art,multi-layered collages,fragmented icons]")
        prompt = self.prompt_builder(oneshot, raw_prompt)
        generated_text = self.model(prompt)[0]["generated_text"]
        concepts = self.extract_answers(generated_text)
        concepts_index = self.answer2index(concepts, raw_prompt)
        return {"concepts":concepts,"concepts_index":concepts_index}
