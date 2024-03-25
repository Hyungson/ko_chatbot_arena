from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ( AzureChatOpenAI,
ChatOpenAI
)
from langchain_mistralai.chat_models import ChatMistralAI

from langchain.prompts import (
    PromptTemplate,
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage,AIMessage

import requests 
import json
import os
import re
from utils import (
    google_api_key,
    azure_openai_api_key_gpt35turbo,
    azure_openai_endpoint_gpt35turbo,
    azure_openai_api_key_gpt4,
    azure_openai_endpoint_gpt4,
    solar_api_key,
    solar_base_url,
    mnc_api_key,
    mistral_api_key,
    clova_api_key,
    clova_gateway_key,
    clova_request_id
    
)


openai_llm = [
    'gpt-35-turbo-1106',
    'gpt4-1106-preview',
]
google_llm = [
    'gemini-pro',
]
mnc_llm = [
    'MOIS-AWQ-240319'
]
solar_llm = [
    'solar-1-mini-chat'
]
mistral_llm = [
    'open-mistral-7b'
]
clova_llm = [
    'HCX-03'
]
llms =  mistral_llm + clova_llm + openai_llm  + mnc_llm  + google_llm  + solar_llm 

conversation_template = """당신은 AI 어시스턴트 입니다.

        # 이전 대화 내용: {history}
        # 사용자 질문: {input}
        # Assistant: """

rag_template = """ 당신은 AI 어시스턴트 입니다. 아래에 제시된 참고 자료를 바탕으로 사용자의 질문에 답을 하세요.:
                    # 참고 자료: {context}

                    # 이전 대화 내용 : {history}
                    # 사용자 질문: {input}
                    # Assistant:"""

class ChatBot():
    
    def __init__(self,model_name,temperature,max_tokens):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history = None
        self.template = conversation_template 

        self.mnc_history=[]
        self.mnc_model = None
        self.hc_history=[]

        if self.model_name in google_llm:

            os.environ["GOOGLE_API_KEY"] = google_api_key
            model = GoogleGenerativeAI(temperature=self.temperature,
                          max_tokens = self.max_tokens,
                          model=self.model_name,
                          convert_system_message_to_human=True)
            template = self.template
            memory=ConversationBufferMemory(ai_prefix="AI Assistant")
            PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
            conversation = ConversationChain(
                prompt=PROMPT,
                llm=model,
                verbose=True,
                memory=memory,
            )
            self.conversation = conversation
            self.history = memory.chat_memory.messages

        elif self.model_name in mistral_llm:
            os.environ["MISTRAL_API_KEY"] = mistral_api_key
            model = ChatMistralAI(model=self.model_name,
                             temperature=self.temperature,
                             max_tokens=self.max_tokens)
            template = self.template
            
            memory = ConversationBufferMemory(ai_prefix="AI Assistant")
            PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
            conversation = ConversationChain(
                prompt=PROMPT,
                llm=model,
                verbose=True,
                memory=memory,
            )
            self.conversation = conversation
            self.history = memory.chat_memory.messages

        elif self.model_name in openai_llm:
            if self.model_name == 'gpt-35-turbo-1106':

                os.environ['AZURE_OPENAI_API_KEY'] = azure_openai_api_key_gpt35turbo
                os.environ['AZURE_OPENAI_ENDPOINT'] =azure_openai_endpoint_gpt35turbo
                api_version = "2024-02-15-preview"
                model = AzureChatOpenAI(openai_api_version=api_version,
                                        azure_deployment = self.model_name,)
                template = self.template
                memory = ConversationBufferMemory(ai_prefix="AI Assistant")
                PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
                conversation = ConversationChain(
                    prompt=PROMPT,
                    llm=model,
                    verbose=True,
                    memory=memory,
                )
                self.conversation = conversation
                self.history = memory.chat_memory.messages

            elif self.model_name == 'gpt4-1106-preview':

                os.environ['AZURE_OPENAI_API_KEY'] = azure_openai_api_key_gpt4
                os.environ['AZURE_OPENAI_ENDPOINT'] = azure_openai_endpoint_gpt4
                api_version = "2024-02-15-preview"
                model = AzureChatOpenAI(openai_api_version=api_version,
                                        azure_deployment = self.model_name,)
                template =self.template
                memory = ConversationBufferMemory(ai_prefix="AI Assistant")
                PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
                conversation = ConversationChain(
                    prompt=PROMPT,
                    llm=model,
                    verbose=True,
                    memory=memory,
                )
                self.conversation = conversation
                self.history = memory.chat_memory.messages

        elif self.model_name in solar_llm:
            model = ChatOpenAI(api_key=solar_api_key,
                               base_url=solar_base_url,
                                  model = self.model_name ,)
            template = self.template
            memory = ConversationBufferMemory(ai_prefix="AI Assistant")
            PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
            conversation = ConversationChain(
                prompt=PROMPT,
                llm=model,
                verbose=True,
                memory=memory,
            )
            self.conversation = conversation
            self.history = memory.chat_memory.messages
            
    def load_demo_single(self,text):
        
        # case of mnc llm 
        if self.model_name in mnc_llm:
            input = text
            template = f"당신은 AI 어시스턴트 입니다.\
                        Current_message: {self.mnc_history} \
                        Human: {input} \
                        Assistant: "

            data = {
                'prompt': template,
                'model': self.model_name,
                'stop': ['\n\n\n'],
                'stream': True,
                'extra': {
                    'prompt' : 'template',
                    'mode' : 'completion'
                }
            }

            res = requests.post(
                'http://114.110.135.189:9020/worker/v1/engines/MOIS-AWQ/completions',
                headers = {'Authorization' : f'Bearer {mnc_api_key}'},
                json=data,

            )
            answer = []
            try:
                if res.encoding is None:
                    res.encoding = 'utf-8-sig'

                for line in res.iter_lines(decode_unicode=True):
                    if line and line != 'data: [DONE]':
                        data = json.loads(line.removeprefix('data: '))
                        answer.append(data['choices'][0]['text'])
                        
            except Exception as e:
                print(f"Error: {e}")
            self.mnc_history.append(HumanMessage(content=f'{input}')) 
            self.mnc_history.append(AIMessage(content=f'{"".join(answer)}'))
            self.history=self.mnc_history
            output = "".join(answer)

        elif self.model_name in clova_llm:

            def extract_content(string):
                match = re.search(r'"content":"([^"]+)"', string)
                
                if match:
                    return match.group(1)
                else:
                    return ""

            system_msg = f"당신은 AI 어시스턴트 입니다. \
                        Current message : {self.hc_history}"
            input=text
            prompt = [{"role" : "system","content":f"{system_msg}"},
                    {"role":"user","content":f"{input}"},
            ]

            headers = {
                    'Content-Type': 'application/json; charset=utf-8',
                    'Accept': 'text/event-stream',
                    'X-NCP-CLOVASTUDIO-API-KEY': clova_api_key,
                    'X-NCP-APIGW-API-KEY': clova_gateway_key,
                    'X-NCP-CLOVASTUDIO-REQUEST-ID':clova_request_id
                }
            request_data = {
                    'messages': prompt,
                    'topP': 0.8,
                    'topK': 0,
                    'maxTokens': 1024,
                    'temperature': 0.7,
                    'repeatPenalty': 5.0,
                    'stopBefore': [],
                    'includeAiFilters': True,
                    'seed': 0
                }

            answer=[]
            completion = requests.post('https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003',headers=headers,json=request_data, stream=True)
            for line in completion.iter_lines():
                    if line:
                        data = line.decode("utf-8")
                        if 'data' in data and '[DONE]' not in data:
                            answer.append(data)


            content_list = [extract_content(string) for string in answer]
            content_list = content_list[:-1]
            self.hc_history.append(HumanMessage(content=f'{input}')) 
            self.hc_history.append(AIMessage(content=f'{"".join(content_list)}'))
            self.history=self.hc_history
            output = "".join(content_list)

        else:

            output = self.conversation.predict(input=text)
 
        return output
    


class RagChatBot():
            
    def __init__(self,model_name,temperature,max_tokens,context):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.history = None
        self.template = rag_template 
        self.context = context

        self.mnc_history=[]
        self.mnc_model = None
        self.hc_history=[]


        if self.model_name in google_llm:

            os.environ["GOOGLE_API_KEY"] = google_api_key
            model = GoogleGenerativeAI(temperature=self.temperature,
                          max_tokens = self.max_tokens,
                          model=self.model_name,
                          convert_system_message_to_human=True)
            memory=ConversationBufferMemory(ai_prefix="AI Assistant")
            PROMPT = PromptTemplate(input_variables=["history", "input"], template=self.template.format(context = self.context,history="{history}",input="{input}"))
            conversation = ConversationChain(
                prompt=PROMPT,
                llm=model,
                verbose=True,
                memory=memory,
            )
            self.conversation = conversation
            self.history = memory.chat_memory.messages

        elif self.model_name in mistral_llm:
            os.environ["MISTRAL_API_KEY"] = mistral_api_key
            model = ChatMistralAI(model=self.model_name,
                             temperature=self.temperature,
                             max_tokens=self.max_tokens)
            
            memory = ConversationBufferMemory(ai_prefix="AI Assistant")
            PROMPT = PromptTemplate(input_variables=["history", "input"], template=self.template.format(context = self.context,history="{history}",input="{input}"))
            conversation = ConversationChain(
                prompt=PROMPT,
                llm=model,
                verbose=True,
                memory=memory,
            )
            self.conversation = conversation
            self.history = memory.chat_memory.messages

        elif self.model_name in openai_llm:
            if self.model_name == 'gpt-35-turbo-1106':

                os.environ['AZURE_OPENAI_API_KEY'] = azure_openai_api_key_gpt35turbo
                os.environ['AZURE_OPENAI_ENDPOINT'] =azure_openai_endpoint_gpt35turbo
                api_version = "2024-02-15-preview"
                model = AzureChatOpenAI(openai_api_version=api_version,
                                        azure_deployment = self.model_name,)
                memory = ConversationBufferMemory(ai_prefix="AI Assistant")
                PROMPT = PromptTemplate(input_variables=["history", "input"], template=self.template.format(context = self.context,history="{history}",input="{input}"))
                conversation = ConversationChain(
                    prompt=PROMPT,
                    llm=model,
                    verbose=True,
                    memory=memory,
                )
                self.conversation = conversation
                self.history = memory.chat_memory.messages

            elif self.model_name == 'gpt4-1106-preview':

                os.environ['AZURE_OPENAI_API_KEY'] = azure_openai_api_key_gpt4
                os.environ['AZURE_OPENAI_ENDPOINT'] = azure_openai_endpoint_gpt4
                api_version = "2024-02-15-preview"
                model = AzureChatOpenAI(openai_api_version=api_version,
                                        azure_deployment = self.model_name,)
                memory = ConversationBufferMemory(ai_prefix="AI Assistant")
                PROMPT = PromptTemplate(input_variables=["history", "input"], template=self.template.format(context = self.context,history="{history}",input="{input}"))
                conversation = ConversationChain(
                    prompt=PROMPT,
                    llm=model,
                    verbose=True,
                    memory=memory,
                )
                self.conversation = conversation
                self.history = memory.chat_memory.messages

        elif self.model_name in solar_llm:
            model = ChatOpenAI(api_key=solar_api_key,
                               base_url=solar_base_url,
                                  model = self.model_name ,)
            memory = ConversationBufferMemory(ai_prefix="AI Assistant")
            PROMPT = PromptTemplate(input_variables=["history", "input"], template=self.template.format(context = self.context,history="{history}",input="{input}"))
            conversation = ConversationChain(
                prompt=PROMPT,
                llm=model,
                verbose=True,
                memory=memory,
            )
            self.conversation = conversation
            self.history = memory.chat_memory.messages

        elif self.model_name in solar_llm:
            model = ChatOpenAI(api_key=solar_api_key,
                               base_url=solar_base_url,
                                  model = self.model_name ,)
            memory = ConversationBufferMemory(ai_prefix="AI Assistant")
            PROMPT = PromptTemplate(input_variables=["history", "input"], template=self.template.format(context = self.context,history="{history}",input="{input}"))
            conversation = ConversationChain(
                prompt=PROMPT,
                llm=model,
                verbose=True,
                memory=memory,
            )
            self.conversation = conversation
            self.history = memory.chat_memory.messages
            
    def load_demo_single(self,text):        
        # case of mnc llm 
        if self.model_name in mnc_llm:
            input = text
            template = f"Human: 당신은 AI 어시스턴트입니다. 아래에 제시된 참고 자료를 바탕으로 사용자의 질문에 답을 하세요.\n\n###참고 자료###\n{self.context}\n\n###이전 대화###\n{self.mnc_history}###사용자 질문###\n{input}\n\n Assistant: "
            data = {
                'prompt': template,
                'model': self.model_name,
                'stop': ['\n\n\n'],
                'stream': True,
                'extra': {
                    'prompt' : 'template',
                    'mode' : 'completion'
                }
            }

            res = requests.post(
                'http://114.110.135.189:9020/worker/v1/engines/MOIS-AWQ/completions',
                headers = {'Authorization' : f'Bearer {mnc_api_key}'},
                json=data
            )
            answer = []
            try:
                if res.encoding is None:
                    res.encoding = 'utf-8-sig'

                for line in res.iter_lines(decode_unicode=True):
                    if line and line != 'data: [DONE]':
                        data = json.loads(line.removeprefix('data: '))
                        answer.append(data['choices'][0]['text'])
                        
            except Exception as e:
                print(f"Error: {e}")
            self.mnc_history.append(HumanMessage(content=f'{input}')) 
            self.mnc_history.append(AIMessage(content=f'{"".join(answer)}'))
            self.history=self.mnc_history
            output = "".join(answer)

        # case of clova
        elif self.model_name in clova_llm:

            def extract_content(string):
                match = re.search(r'"content":"([^"]+)"', string)
                if match:
                
                    return match.group(1)
                else:
                    return ""

            system_msg = f"당신은 AI 어시스턴트 입니다. \
                        아래에 제시된 참고 자료를 바탕으로 사용자의 질문에 답을 하세요.\n\n###참고 자료###\n{self.context}\
                        Current message : {self.hc_history}"
            input=text
            prompt = [{"role" : "system","content":f"{system_msg}"},
                    {"role":"user","content":f"{input}"},
            ]

            headers = {
                    'Content-Type': 'application/json; charset=utf-8',
                    'Accept': 'text/event-stream',
                    'X-NCP-CLOVASTUDIO-API-KEY': clova_api_key,
                    'X-NCP-APIGW-API-KEY': clova_gateway_key,
                    'X-NCP-CLOVASTUDIO-REQUEST-ID': clova_request_id
                }
            request_data = {
                    'messages': prompt,
                    'topP': 0.8,
                    'topK': 0,
                    'maxTokens': 1024,
                    'temperature': 0.7,
                    'repeatPenalty': 5.0,
                    'stopBefore': [],
                    'includeAiFilters': True,
                    'seed': 0
                }

            answer=[]
            completion = requests.post('https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003',headers=headers,json=request_data, stream=True)
            for line in completion.iter_lines():
                    if line:
                        data = line.decode("utf-8")
                        if 'data' in data and '[DONE]' not in data:
                            answer.append(data)


            content_list = [extract_content(string) for string in answer]
            content_list = content_list[:-1]
            self.hc_history.append(HumanMessage(content=f'{input}')) 
            self.hc_history.append(AIMessage(content=f'{"".join(content_list)}'))
            self.history=self.hc_history
            output = "".join(content_list)

        else:

            output = self.conversation.predict(input=text)
 
        return output
    

    

  


