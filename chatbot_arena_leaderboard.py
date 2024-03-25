from gradio_app_fn import (get_leaderboard, 
                           regen_chat_leaderboard, 
                           regen_rag_leaderboard,
                           save_chat_path,
                           save_rag_path)

import gradio as gr



with gr.Blocks() as leaderboard_demo:
    
    with gr.Column():
        gr.Markdown("# KO Chatbot Arena Leaderboard")
        chat_leaderboard = gr.Dataframe(get_leaderboard(save_chat_path))
        chat_regen_bttn = gr.Button("리더보드 갱신")

        chat_regen_bttn.click(regen_chat_leaderboard, inputs=[],
                            outputs = [chat_leaderboard])

    with gr.Column():
        gr.Markdown("# KO Chatbot Arena RAG Leaderboard")
        rag_leaderboard = gr.Dataframe(get_leaderboard(save_rag_path))
        rag_regen_bttn = gr.Button("리더보드 갱신")

        rag_regen_bttn.click(regen_rag_leaderboard, inputs=[],
                            outputs = [rag_leaderboard])






