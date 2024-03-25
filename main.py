from rag_arena import rag_demo
from chat_arena import arena_demo
from chatbot_arena_leaderboard import leaderboard_demo

import gradio as gr
import argparse

                                    
demo = gr.TabbedInterface([arena_demo,rag_demo ,leaderboard_demo],["Chatbot Arena","RAG Arena" ,"Leaderboard"], title="KO Chatbot Arena Demo")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=21012)

    args = parser.parse_args()

    demo.queue(
        default_concurrency_limit = 10,
        status_update_rate=10,
        api_open=False,

    )
    demo.launch(
        server_name=args.host,
        server_port = args.port,
        max_threads = 200,
        share=False,
        root_path="/ko_llm_arena"
                )
