from utils import json_path
from gradio_app_fn import get_leaderboard

import gradio as gr


save_path = json_path + 'rag_test.json'



with gr.Blocks() as rag_leaderboard_demo:
    gr.Markdown("# KO Chatbot Arena RAG Leaderboard")
    rag_leaderboard = gr.Dataframe(get_leaderboard(save_path))
    rag_regen_bttn = gr.Button("리더보드 갱신")

    rag_regen_bttn.click(regen_leaderboard, inputs=[save_path],
                          outputs = [chat_leaderboard])








