import gradio as gr
import time
import asyncio

lst = []


class A():
    def __init__(self) -> None:
        self.a = []
        self.b = []

    def add_a(self):
        time.sleep(3)
        self.a.append()

with gr.Blocks() as demo:
    btn = gr.Button("Button")

    @btn.click
    def on_button_click():
        asyncio.run(add())

demo.launch()
