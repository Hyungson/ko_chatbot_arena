from llm_api import *
from chatbot_arena_leaderboard import *
from gradio_app_fn import *

import gradio as gr


#def make_arena_demo():

save_path = json_path + 'test.json'

notice_markdown = """ 
# ⚔️🧨 KO Chatbot Arena Demo 🥊⚔️
"""
rule_markdown = """
무작위로 익명의 LLM 모델 두개와 대화를 나눕니다. 대화를 나눈 후, 투표를 할 수 있으며, 투표 결과는 랭킹에 반영됩니다. \n\n
새 게임을 누를 시, 대화내용과 모델이 초기화 됩니다.
대화 초기화를 누를 시, 대화내용만 초기화 됩니다.

투표는 두 모델의 대화가 모두 완료 된 후 진행해야 결과에 반영 됩니다.
"""
with gr.Blocks() as arena_demo:
    
    # init chat_model, chat
    save_path_state = gr.State(save_path)
        
    model_name_state1 = gr.State([])
    model_name_state2 = gr.State([])

    model_chat_state1 = gr.State([])    
    model_chat_state2 = gr.State([])
    
    # load 2 models at name_states
    arena_demo.load(init,outputs=[model_name_state1,model_name_state2]) 
    
    # title
    with gr.Row():
        gr.Markdown(notice_markdown)

    with gr.Accordion("📜 안내 사항 보기📜",open=False):
        gr.Markdown(rule_markdown)

    # chatbot
                
    with gr.Row() :
        for i in range(1,3):
            label = "Model A" if i == 1 else "Model B"
            with gr.Column():
                globals()['_chatbot' + str(i)] = gr.Chatbot(label=label)
    
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
    for i in range(1,3):        
        msg.submit(user,[msg, globals()['_chatbot' + str(i)]],[msg, globals()['_chatbot' + str(i)]],queue=False).then(
            bot, [ globals()[f'model_name_state{str(i)}'], globals()['_chatbot' + str(i)]], globals()['_chatbot' + str(i)]).then(
            save_chat_state, [globals()['_chatbot' + str(i)]],[globals()[f'model_chat_state' + str(i)]]).then(
                activate_buttons, inputs=None,outputs= [voteA_button, voteB_button, vote_tie_button, both_bad_button])
        
        send_button.click(user,[msg, globals()['_chatbot' + str(i)]],[msg, globals()['_chatbot' + str(i)]],queue=False).then(
            bot, [ globals()[f'model_name_state{str(i)}'], globals()['_chatbot' + str(i)]], globals()['_chatbot' + str(i)]).then(
            save_chat_state, [globals()['_chatbot' + str(i)]],[globals()[f'model_chat_state' + str(i)]]).then(
                activate_buttons, inputs=None,outputs= [voteA_button, voteB_button, vote_tie_button, both_bad_button])
            
    # button events
        
    vote_bttn_inputs = [model_name_state1,model_name_state2,model_chat_state1,save_path_state]
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
            new_round_button.click(new_round, inputs = [],
                        outputs=[model_name_state1,model_name_state2, model_chat_state1,model_chat_state2,
                                msg,_chatbot1,_chatbot2, show_A_name,show_B_name,
                                voteA_button, voteB_button, vote_tie_button, both_bad_button])
        with gr.Column():
            regenerate_button = gr.Button("🌧️ 대화 초기화 🌧️")
            regenerate_button.click(lambda : [None] * 3, outputs = [msg,_chatbot1,_chatbot2] )

#return arena_demo