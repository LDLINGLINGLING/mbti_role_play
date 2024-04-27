from transformers import AutoModelForCausalLM, AutoTokenizer,GenerationConfig
from abc import ABC
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from peft import PeftModel
import torch
import gradio as gr


def load_model(path="/ai/ld/pretrain/Qwen1.5-4B/qwen/Qwen1___5-4B/",lora_path='/ai/ld/remote/LLaMA-Factory-main/mbti_model/checkpoint-1000',eos_id=[151643,151645]):
    # the device to load the model onto

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    # 加载基础模型
    max_memory = f"{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB"
    generation_config = GenerationConfig.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="cuda:0",
                max_memory=max_memory,
                trust_remote_code=True,
                #use_safetensors=True,
                #bf16=True
            ).eval()
    # stop_words=['\n\n\n', 'Action','Final']
    # stop_words_ids=[tokenizer.encode(w) for w in stop_words]+[[151643]]
    model.generation_config.eos_token_id=[151643,151645]
    #model.generation_config.top_k = 1
    if lora_path:
        # 加载PEFT模型
        model = PeftModel.from_pretrained(model, lora_path)

    return model,tokenizer

model,tokenizer=load_model()
device='cuda'

class Qwen(LLM, ABC):
     max_token: int = 10000
     temperature: float = 0.01
     top_p = 0.9
     history_len: int = 3

     def __init__(self):
         super().__init__()
         
     @property
     def _llm_type(self) -> str:
         return "Qwen"

     @property
     def _history_len(self) -> int:
         return self.history_len

     def set_history_len(self, history_len: int = 10) -> None:
         self.history_len = history_len

     def _call(
         self,
         prompt: str,
         stop: Optional[List[str]] = None,
         run_manager: Optional[CallbackManagerForLLMRun] = None,
     ) -> str:
         messages = [
             {"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": prompt}
         ]
         text = tokenizer.apply_chat_template(
             messages,
             tokenize=False,
             add_generation_prompt=True
         )
         model_inputs = tokenizer([text], return_tensors="pt").to(device)
         generated_ids = model.generate(
             model_inputs.input_ids,
             max_new_tokens=512
         )
         generated_ids = [
             output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
         ]

         response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
         return response

     @property
     def _identifying_params(self) -> Mapping[str, Any]:
         """Get the identifying parameters."""
         return {"max_token": self.max_token,
                 "temperature": self.temperature,
                 "top_p": self.top_p,
                 "history_len": self.history_len}
if __name__=='__main__':
    q=Qwen()
    type_dict={'I':'内向的','E':'外向的','S':'感官的','N':'直觉的','T':'理性的','F':'感性的','J':'判断的','P':'感知的'}
    while True:
        question=input('请输入问题')
        if question=='exit':
            break
        if question[0:4].upper() in ['INTJ','ENFP','ENTP','ESTP','ISTP','ISFP','INFP','ENTJ','ESTJ','INTP','ISTJ','INFJ','ENFJ','ENTJ','ESFP','ESFJ','ISFJ'] :
            type_merge=[]
            for i in question[0:4].upper():
                type_merge.append(type_dict[i])
                prompt='请你作为一个{}人，回答以下问题：'.format('、'.join(type_merge))
                question=prompt+question[4:]
        
        print(q._call(question))
    def gradio_response(question, history):
        try:
            global history_copy,messages,pre_router
            if question=='clear':
                history=[]
            if question[0:4].upper() in ['INTJ','ENFP','ENTP','ESTP','ISTP','ISFP','INFP','ENTJ','ESTJ','INTP','ISTJ','INFJ','ENFJ','ENTJ','ESFP','ESFJ','ISFJ'] :
                type_merge=[]
                for i in question[0:4].upper():
                    type_merge.append(type_dict[i])
                    prompt='请你作为一个{}人，回答以下问题：'.format('、'.join(type_merge))
                    question=prompt+question[4:]
            return q._call(question)
        except Exception as e:
            return '出现错误{},请重新输入问题'.format(str(e))
    demo = gr.ChatInterface(gradio_response,chatbot=gr.Chatbot(label="Chagent:\nmbti_chat_bot",height=700))
    demo.launch(share=True)