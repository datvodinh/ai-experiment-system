import gradio as gr
import requests
import trainer as tr
import pandas as pd
from itertools import product
headers = ["lr", "batch_size", "max_epochs", "dropout"]


def generate():
    # url = "http://host.docker.internal:8000/info/"
    # response = requests.get(url)
    # return response.json()
    return pd.DataFrame(
        {
            "lr": [0.0001, 0.01, 0.00001, 0.1],
            "batch_size": [32, 32, 64, 64],
            "max_epochs": [50, 50, 30, 30],
            "dropout": [0.1, 0, 0, 0]
        }
    )


def get_grid_seach(lr, batch_size, max_epochs, dropout):
    dct = {
        "lr": lr,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "dropout": dropout
    }
    keys = list(dct.keys())
    df = pd.DataFrame(
        [
            dict(zip(keys, cv)) for cv in product(*list(dct.values()))
        ],
        columns=keys
    )
    return df


def get_statistic():
    return pd.DataFrame(
        {
            "status": ["done", "done", "done", "done"],
            "accuracy": [0.99, 0.95, 0.98, 0.96],
            "lr": [0.0001, 0.01, 0.00001, 0.1],
            "batch_size": [32, 32, 64, 64],
            "max_epochs": [50, 50, 30, 30],
            "dropout": [0.1, 0, 0, 0]
        }
    )


with gr.Blocks(theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown("# Deep Learning Experiment Management System")
    with gr.Column():

        gr.Markdown("## Hyperparameters")
        with gr.Row(variant='panel'):
            with gr.Column(variant='panel'):
                lr = gr.CheckboxGroup(
                    label="Learning Rate",
                    choices=[0.01, 0.001, 0.0001, 0.00001],
                    value=0.0001, interactive=True
                )
                batch_size = gr.CheckboxGroup(
                    label="Batch Size",
                    choices=[32, 64, 128, 256],
                    value=32, interactive=True
                )
            with gr.Column(variant='panel'):
                max_epochs = gr.CheckboxGroup(
                    label="Max Epochs",
                    choices=[10, 20, 30, 40],
                    value=20, interactive=True
                )
                dropout = gr.CheckboxGroup(
                    label="Dropout",
                    choices=[0, 0.1, 0.2, 0.3],
                    value=0., interactive=True
                )

        with gr.Row(variant='panel'):
            train_btn = gr.Button("Add Job")
            reset_btn = gr.Button("Reset Params")
            reload_btn = gr.Button("Reload Data")

        gr.Markdown("## Running")
        with gr.Column(variant='panel'):
            progress = gr.Textbox(label="Progress", value="Idle", interactive=False)
            with gr.Row(variant='panel'):
                job_running = gr.DataFrame(label="Jobs Running", headers=headers)
                job_queuing = gr.DataFrame(label="Jobs in Queue", headers=headers)

        gr.Markdown("## Statistics")
        with gr.Row(variant='compact'):

            job_finish = gr.DataFrame(label="Jobs Finished", headers=["status", "accuracy"] + headers)

    train_btn.click(get_grid_seach, inputs=[lr, batch_size, max_epochs, dropout], outputs=job_queuing)
    reset_btn.click(lambda: [0.001, 256, 10, 0], outputs=[lr, batch_size, max_epochs, dropout])
    # test_btn.click(print_test, inputs=[lr])

demo.launch(share=False, server_name="0.0.0.0")
