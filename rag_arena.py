#from utils import json_path
from chatbot_arena_leaderboard import *
from gradio_app_fn import *
import gradio as gr
from gradio_app_fn import *
from llm_api import *


#def rag_arena():
save_path = json_path + 'rag_test.json'

notice_markdown = """ 
# ⚔️🧨 KO Chatbot RAG Arena Demo 🥊⚔️
"""
rule_markdown = """
무작위로 익명의 LLM 모델 두개와 대화를 나눕니다. 대화를 나눈 후, 투표를 할 수 있으며, 투표 결과는 랭킹에 반영됩니다.\n\n
모델의 프롬프트에 반영될 context를 입력하세요.
모델은 context에 기반해 대화를 생성합니다.\n\n
새 게임을 누를 시, 대화내용과 모델이 초기화 됩니다.
대화 초기화를 누를 시, 대화내용만 초기화 됩니다.

투표는 두 모델의 대화가 모두 완료 된 후 진행해야 결과에 반영 됩니다.
"""

with gr.Blocks() as rag_demo:
    
    # init chat_model, chat

    save_path_state = gr.State(save_path)
        
    rag_model_name_state1 = gr.State([])
    rag_model_name_state2 = gr.State([])

    rag_model_chat_state1 = gr.State([])
    rag_model_chat_state2 = gr.State([])
    
    # load 2 models at name_states
    
    
    # title
    with gr.Row():
        gr.Markdown(notice_markdown)
    
    with gr.Accordion("📜 안내 사항 보기📜",open = False):
        gr.Markdown(rule_markdown)

    # chatbot
                
    with gr.Row() :
        context_area = gr.TextArea(label="Context",interactive=False)
    with gr.Row() :    
        context_box = gr.Textbox(label="Context를 입력하세요.")
        context_send_button = gr.Button("입력",scale=0)
    with gr.Row() :
        for i in range(1,3):
            label = "Model A" if i == 1  else "ModelB" 
            globals()['_rag_chatbot' + str(i)] = gr.Chatbot(label=label)

    
    # buttons
    with gr.Row():
        with gr.Column():
            voteA_button = gr.Button("A에게 투표 👈", visible=False)

        with gr.Column():
            voteB_button = gr.Button("B에게 투표 👉",visible=False)

        with gr.Column():
            vote_tie_button = gr.Button("명승부 👏",visible=False)
        
        with gr.Column():
            both_bad_button = gr.Button("두 모델 모두 불만족 👎",visible=False)

    with gr.Row():
        with gr.Column():
            show_A_name = gr.Markdown()
        with gr.Column():
            show_B_name = gr.Markdown()

    # chat input 
    
            
    with gr.Row():
        msg = gr.Textbox(label="메시지를 입력하세요.")
        send_button = gr.Button("입력",scale=0)

    context_box.submit(set_context,[context_box], [context_area,context_box]).then(rag_init,inputs=[context_area],outputs=[rag_model_name_state1,rag_model_name_state2])
    context_send_button.click(set_context,[context_box], [context_area,context_box]).then(rag_init,inputs=[context_area],outputs=[rag_model_name_state1,rag_model_name_state2])

    for i in range(1,3):        
        msg.submit(user,[msg, globals()['_rag_chatbot' + str(i)]],[msg, globals()['_rag_chatbot' + str(i)]],queue=False).then(
            bot, [ globals()[f'rag_model_name_state{str(i)}'], globals()['_rag_chatbot' + str(i)]], globals()['_rag_chatbot' + str(i)]).then(
            save_chat_state, [globals()['_rag_chatbot' + str(i)]],[globals()[f'rag_model_chat_state' + str(i)]]).then(
                activate_buttons, inputs=None,outputs= [voteA_button, voteB_button, vote_tie_button, both_bad_button])
        
        send_button.click(user,[msg, globals()['_rag_chatbot' + str(i)]],[msg, globals()['_rag_chatbot' + str(i)]],queue=False).then(
            bot, [ globals()[f'rag_model_name_state{str(i)}'], globals()['_rag_chatbot' + str(i)]], globals()['_rag_chatbot' + str(i)]).then(
            save_chat_state, [globals()['_rag_chatbot' + str(i)]],[globals()[f'rag_model_chat_state' + str(i)]]).then(
                activate_buttons, inputs=None,outputs= [voteA_button, voteB_button, vote_tie_button, both_bad_button])
            
    # button events
        
    vote_bttn_inputs = [rag_model_name_state1,rag_model_name_state2,rag_model_chat_state1,save_path_state,context_area]
    vote_bttn_outputs = [show_A_name,show_B_name, voteA_button, voteB_button, vote_tie_button, both_bad_button]


    voteA_button.click(voteA, inputs=vote_bttn_inputs,
                            outputs=vote_bttn_outputs)
    voteB_button.click(voteB, inputs=vote_bttn_inputs,
                            outputs=vote_bttn_outputs)
    vote_tie_button.click(VoteTie, inputs=vote_bttn_inputs,
                                outputs=vote_bttn_outputs)
    both_bad_button.click(both_bad, inputs=vote_bttn_inputs,
                                outputs=vote_bttn_outputs)
                            
    # new round,regenerate button

    with gr.Row():
        with gr.Column():     
            new_round_button = gr.Button("🔄 새 게임 🔄")
            new_round_button.click(rag_new_round_1,inputs = [],outputs =[context_box,context_area]).then(rag_new_round_2, inputs = [context_area],
                        outputs=[rag_model_name_state1,rag_model_name_state2, rag_model_chat_state1,rag_model_chat_state2,
                                msg,_rag_chatbot1,_rag_chatbot2, show_A_name,show_B_name,
                                voteA_button, voteB_button, vote_tie_button, both_bad_button])
        with gr.Column():
            regenerate_button = gr.Button("🌧️ 대화 초기화 🌧️")
            regenerate_button.click(lambda : [None] * 3, outputs = [msg,_rag_chatbot1,_rag_chatbot2] )
    
#return rag_demo

