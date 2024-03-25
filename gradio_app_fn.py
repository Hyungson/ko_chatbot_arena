from llm_api import *
from utils import json_path
import numpy as np
import random
import os
import json
import time
import gradio as gr
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
pd.options.display.float_format = '{:.2f}'.format


save_chat_path = json_path + 'test.json'
save_rag_path = json_path + 'rag_test.json'

# entire llm models list
def get_model_list(llms=llms):
    models = random.sample(llms,2)
    return models

def set_context(context):
    return context, None

# chatbot respond
## save chat history

def user(message,history):
    return "", history + [[message, None]]

def bot(model_name_state, history):
    bot_messsage = model_name_state.load_demo_single(history[-1][0])
    history[-1][1] = ""
    for character in bot_messsage:
        history[-1][1] += character
        time.sleep(0.01)
        yield history
 
def save_chat_state(_chatbot1):
    return _chatbot1

def activate_buttons():
    return gr.Button(visible=True,interactive=True),gr.Button(visible=True,interactive=True),\
        gr.Button(visible=True,interactive=True),gr.Button(visible=True,interactive=True)


## set model name to model_name_state1, model_name_state2
def rag_init(context_box):
    global llms
    
    models = get_model_list(llms)
    chatbot1 = RagChatBot(model_name=models[0],temperature=0.7,max_tokens=1024, context = context_box)
    chatbot2 = RagChatBot(model_name=models[1],temperature=0.7,max_tokens=1024, context = context_box)

    return chatbot1, chatbot2

# reset everything

def rag_new_round_1():
    return None, None#gr.Textbox(label="Context를 입력하세요."), gr.TextArea(label="Context")


def rag_new_round_2(context_area):
    
    global llms
    
    models = get_model_list(llms)
    chatbot1 = RagChatBot(model_name=models[0],temperature=0.7,max_tokens=1024,context=context_area)
    chatbot2 = RagChatBot(model_name=models[1],temperature=0.7,max_tokens=1024, context=context_area)

    return  chatbot1,  chatbot2, gr.State([]), gr.State([]), \
        None, None, None,gr.Markdown(visible=False),gr.Markdown(visible=False)\
        ,gr.Button(visible=False),gr.Button(visible=False),gr.Button(visible=False),gr.Button(visible=False)

# voting functions
## after voting, save the data to json file, show models, and disable buttons

## set model name to model_name_state1, model_name_state2
def init():
    global llms, model_name_state1, model_name_state2 ,  _chatbot1, _chatbot2, msg
    
    models = get_model_list(llms)
    chatbot1 = ChatBot(model_name=models[0],temperature=0.7,max_tokens=1024)
    chatbot2 = ChatBot(model_name=models[1],temperature=0.7,max_tokens=1024)

    return chatbot1, chatbot2

# new round function
## reset everything

def new_round():
    
    global llms
    
    models = get_model_list(llms)
    chatbot1 = ChatBot(model_name=models[0],temperature=0.7,max_tokens=1024)
    chatbot2 = ChatBot(model_name=models[1],temperature=0.7,max_tokens=1024)

    return  chatbot1,  chatbot2, gr.State([]), gr.State([]), \
        None, None, None,gr.Markdown(visible=False),gr.Markdown(visible=False)\
        ,gr.Button(visible=False),gr.Button(visible=False),gr.Button(visible=False),gr.Button(visible=False)




def voteA(model_name_state1,model_name_state2,model_chat_state1,request : gr.Request,save_path,context=None):
    if save_path == save_chat_path:
        save_dict = {
            "IP" : f"{request.client.host}",
            "winner" : "model_a",
            "turn" : len(model_chat_state1),
            "model_a": f"{model_name_state1.model_name}",
            "model_b" : f"{model_name_state2.model_name}",
            "chats_a" : f"{model_name_state1.history}",
            "chats_b" : f"{model_name_state2.history}",
            "time_stamp" : time.time()

        }
    else:
        save_dict ={
            "IP" : f"{request.client.host}",
            "winner" : "model_a",
            "turn" : len(model_chat_state1),
            "model_a": f"{model_name_state1.model_name}",
            "model_b" : f"{model_name_state2.model_name}",
            "context" : f"{context}",
            "chats_a" : f"{model_name_state1.history}",
            "chats_b" : f"{model_name_state2.history}",
            "time_stamp" : time.time()

        }
    # test.json
    if os.path.exists(save_path):
        with open(save_path,'r', encoding='utf-8') as outfile:
            data = json.load(outfile)
        data['chats'].append(save_dict)
        with open(save_path,'w', encoding='utf-8') as outfile:
            json.dump(data,outfile,indent='\t',ensure_ascii=False)
    else:
        data = {}
        data['chats'] = []
        data['chats'].append(save_dict)
        with open(save_path,'a', encoding='utf-8') as outfile:
            data = json.dump(data,outfile,indent='\t',ensure_ascii=False)
    

    return gr.Markdown(f'### Model A : {model_name_state1.model_name}',visible=True),gr.Markdown(f'### Model B : {model_name_state2.model_name}',visible=True) , \
        gr.Button(interactive=False),gr.Button(interactive=False),\
        gr.Button(interactive=False),gr.Button(interactive=False)



def voteB(model_name_state1,model_name_state2,model_chat_state1,request : gr.Request,save_path,context=None):
    if save_path == save_chat_path:
        save_dict ={
            "IP" : f"{request.client.host}",
            "winner" : "model_b",
            "turn" : len(model_chat_state1),
            "model_a": f"{model_name_state1.model_name}",
            "model_b" : f"{model_name_state2.model_name}",
            "chats_a" : f"{model_name_state1.history}",
            "chats_b" : f"{model_name_state2.history}",
            "time_stamp" : time.time()

        }
    else:
        save_dict ={
            "IP" : f"{request.client.host}",
            "winner" : "model_b",
            "turn" : len(model_chat_state1),
            "model_a": f"{model_name_state1.model_name}",
            "model_b" : f"{model_name_state2.model_name}",
            "context" : f"{context}",
            "chats_a" : f"{model_name_state1.history}",
            "chats_b" : f"{model_name_state2.history}",
            "time_stamp" : time.time()

        }
    # test.json
    if os.path.exists(save_path):
        with open(save_path,'r', encoding='utf-8') as outfile:
            data = json.load(outfile)
        data['chats'].append(save_dict)
        with open(save_path,'w', encoding='utf-8') as outfile:
            json.dump(data,outfile,indent='\t',ensure_ascii=False)
    else:
        data = {}
        data['chats'] = []
        data['chats'].append(save_dict)
        with open(save_path,'a', encoding='utf-8') as outfile:
            data = json.dump(data,outfile,indent='\t',ensure_ascii=False)
    


    return gr.Markdown(f'### Model A : {model_name_state1.model_name}',visible=True),gr.Markdown(f'### Model B : {model_name_state2.model_name}',visible=True) , \
        gr.Button(interactive=False),gr.Button(interactive=False),\
        gr.Button(interactive=False),gr.Button(interactive=False)

def VoteTie(model_name_state1,model_name_state2,model_chat_state1,request : gr.Request,save_path,context=None):
    if save_path == save_chat_path:
        save_dict ={
            "IP" : f"{request.client.host}",
            "winner" : "tie",
            "turn" : len(model_chat_state1),
            "model_a": f"{model_name_state1.model_name}",
            "model_b" : f"{model_name_state2.model_name}",
            "chats_a" : f"{model_name_state1.history}",
            "chats_b" : f"{model_name_state2.history}",
            "time_stamp" : time.time()
        }

        
    else:
        save_dict ={
            "IP" : f"{request.client.host}",
            "winner" : "tie",
            "turn" : len(model_chat_state1),
            "model_a": f"{model_name_state1.model_name}",
            "model_b" : f"{model_name_state2.model_name}",
            "context" : f"{context}",
            "chats_a" : f"{model_name_state1.history}",
            "chats_b" : f"{model_name_state2.history}",
            "time_stamp" : time.time()
        }

    if os.path.exists(save_path):
            with open(save_path,'r', encoding='utf-8') as outfile:
                data = json.load(outfile)
            data['chats'].append(save_dict)
            with open(save_path,'w', encoding='utf-8') as outfile:
                json.dump(data,outfile,indent='\t',ensure_ascii=False)
    else:
        data = {}
        data['chats'] = []
        data['chats'].append(save_dict)
        with open(save_path,'a', encoding='utf-8') as outfile:
            data = json.dump(data,outfile,indent='\t',ensure_ascii=False)

    return gr.Markdown(f'### Model A : {model_name_state1.model_name}',visible=True),gr.Markdown(f'### Model B : {model_name_state2.model_name}',visible=True) , \
        gr.Button(interactive=False),gr.Button(interactive=False),\
        gr.Button(interactive=False),gr.Button(interactive=False)

def both_bad(model_name_state1,model_name_state2,model_chat_state1,request : gr.Request,save_path,context=None):
    if save_path == save_chat_path:
        save_dict ={
            "IP" : f"{request.client.host}",
            "winner" : "tie(both bad)",
            "turn" : len(model_chat_state1),
            "model_a": f"{model_name_state1.model_name}",
            "model_b" : f"{model_name_state2.model_name}",
            "chats_a" : f"{model_name_state1.history}",
            "chats_b" : f"{model_name_state2.history}",
            "time_stamp" : time.time()
        }
    else:
        save_dict ={
            "IP" : f"{request.client.host}",
            "winner" : "tie(both bad)",
            "turn" : len(model_chat_state1),
            "model_a": f"{model_name_state1.model_name}",
            "model_b" : f"{model_name_state2.model_name}",
            "context" : f"{context}",
            "chats_a" : f"{model_name_state1.history}",
            "chats_b" : f"{model_name_state2.history}",
            "time_stamp" : time.time()
        }
    if os.path.exists(save_path):
        with open(save_path,'r', encoding='utf-8') as outfile:
            data = json.load(outfile)
        data['chats'].append(save_dict)
        with open(save_path,'w', encoding='utf-8') as outfile:
            json.dump(data,outfile,indent='\t',ensure_ascii=False)
    else:
        data = {}
        data['chats'] = []
        data['chats'].append(save_dict)
        with open(save_path,'a', encoding='utf-8') as outfile:
            data = json.dump(data,outfile,indent='\t',ensure_ascii=False)


    return gr.Markdown(f'### Model A : {model_name_state1.model_name}',visible=True),gr.Markdown(f'### Model B : {model_name_state2.model_name}',visible=True) , \
        gr.Button(interactive=False),gr.Button(interactive=False),\
        gr.Button(interactive=False),gr.Button(interactive=False)

def get_battle_df(save_path):
    with open(save_path,'r', encoding='utf-8') as outfile:
            data = json.load(outfile)
    raw_data = pd.DataFrame(data['chats']).sort_values(ascending=True, by=["time_stamp"])
    raw_data.drop(columns='IP',inplace=True)
    battles = raw_data
    return battles

def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    rating = defaultdict(lambda: INIT_RATING)

    for rd, model_a, model_b, winner in battles[['model_a', 'model_b', 'winner']].itertuples():
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if winner == "model_a":
            sa = 1
        elif winner == "model_b":
            sa = 0
        elif winner == "tie" or winner == "tie(both bad)":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {winner}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)

    return rating

def compute_voting(battles):
    voting = defaultdict(lambda: 0)
    for rd, model_a, model_b, winner in battles[['model_a', 'model_b', 'winner']].itertuples():
        
        if winner == "model_a":
            voting[model_a] += 1
        elif winner == "model_b":
            voting[model_b] += 1
        elif winner == "tie" or winner == "tie(both bad)":
            voting[model_a] += 0.5
            voting[model_b] += 0.5
        else:
            raise Exception(f"unexpected vote {winner}")

    return voting

def get_bootstrap_result(battles, func_compute_elo, num_round):
    rows = []
    for i in tqdm(range(num_round), desc="bootstrap"):
        rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True)))
    df = pd.DataFrame(rows)

    return df[df.median().sort_values(ascending=False).index]

def compute_bootstrap_ci_scores(df):
    bars = pd.DataFrame(dict(
        lower = df.quantile(.025),
        rating = df.quantile(.5),
        upper = df.quantile(.975))).reset_index(names="Model").sort_values("rating", ascending=False)
    bars['error_y'] = bars['upper'] - bars["rating"]
    bars['error_y_minus'] = bars['rating'] - bars["lower"]
    bars['error_y'] = bars['error_y'].apply(lambda x: str(int(x)))
    bars['error_y_minus'] = bars['error_y_minus'].apply(lambda x: str(int(x)))

    return bars

def get_leaderboard(save_path):

    battles = get_battle_df(save_path)
    votings = compute_voting(battles)

    BOOTSTRAP_ROUNDS = 1000
    np.random.seed(42)
    bootstrap_elo_lu = get_bootstrap_result(battles, compute_elo, BOOTSTRAP_ROUNDS)
    bootstrap_lu_median = bootstrap_elo_lu.median().reset_index().set_axis(["Model", "Elo rating"], axis=1)
    bootstrap_lu_median["Elo rating"] = (bootstrap_lu_median["Elo rating"] + 0.5).astype(int)
    bootstrap_lu_median.index = bootstrap_lu_median.index + 1
    bootstrap_lu_median.insert(0, "Rank", bootstrap_lu_median.index)
    bootstrap_lu_median["Votes"] = bootstrap_lu_median["Model"].map(votings)

    bars = compute_bootstrap_ci_scores(bootstrap_elo_lu).sort_values("Model")
    df = pd.merge(bootstrap_lu_median, bars, on="Model")
    df["95% CI"] = "+" +df["error_y"] + " / -" + df["error_y_minus"]

    df.drop(columns=["lower", "upper","rating",'error_y','error_y_minus'], inplace=True)
    

    return df

def regen_rag_leaderboard():
    df = get_leaderboard(save_rag_path)
    return df

def regen_chat_leaderboard():
    df = get_leaderboard(save_chat_path)
    return df



    