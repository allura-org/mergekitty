# Copyright (C) 2025 Allura-org

try:
    import gradio as gr
    from gradio_log import Log
    from gradio_huggingfacehub_search import HuggingfaceHubSearch
except ImportError:
    raise ImportError(
        "gradio is not installed. Please install mergekitty with `pip install mergekitty[gradio]`."
    )
from mergekitty.merge import MergeConfiguration, MergeOptions, run_merge
from mergekitty.merge_methods import REGISTERED_MERGE_METHODS
import torch
import yaml
import huggingface_hub as hf
import click
import tempfile
from contextlib import redirect_stdout, redirect_stderr

from mergekitty.merge_methods.base import MergeMethod

PROD = True


def do_merge(
    token: gr.OAuthToken, yaml_config: str, log_file: str, output_model_path: str
):
    config = MergeConfiguration.model_validate(yaml.safe_load(yaml_config))
    options = MergeOptions(
        allow_crimes=False,
        allow_remote_code=True,
        lazy_unpickle=True,
        cuda=torch.cuda.is_available(),
    )
    with open(log_file, "w") as f:
        with redirect_stdout(f), redirect_stderr(f):
            temp_dir = tempfile.mkdtemp()
            run_merge(config, temp_dir, options)
            hf.create_repo(output_model_path, repo_type="model", private=True)
            hf.upload_folder(
                repo_id=output_model_path,
                folder_path=temp_dir,
                repo_type="model",
                token=token.token,
            )
            gr.Info(
                f"Model uploaded to <a href='https://huggingface.co/{output_model_path}'>{output_model_path}</a> successfully!"
            )
    with open(log_file, "w") as f:
        f.write("Merge complete!")
    return


def main(share: bool):
    tmp = tempfile.NamedTemporaryFile(delete=False)
    with gr.Blocks() as gui:
        state = gr.State(value=tmp.name)
        gr.Markdown("# Mergekitty GUI")
        with gr.Tab("YAML config"):
            with gr.Row():
                with gr.Column():
                    output_model_path = gr.Textbox(
                        label="Output model path (Huggingface identifier, ie `estrogen/test-model` -- this should be a user or org you have write access to)",
                        value="estrogen/test-model",
                    )
                    yaml_config = gr.Code(
                        language="yaml",
                    )
                    btn_yaml = gr.Button("Merge!")

                with gr.Column():
                    Log(log_file=tmp.name, dark=True, label="Logs", container=True)
        with gr.Tab("GUI config"):
            if PROD:
                with gr.Row():
                    gr.Markdown("TODO!")
            else:
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            output_model_path = gr.Textbox(
                                label="Output model path (Huggingface identifier, ie `estrogen/test-model` -- this should be a user or org you have write access to)",
                                value="estrogen/test-model",
                            )
                            HuggingfaceHubSearch(label="Base model", interactive=True)
                            merge_method = gr.Dropdown(
                                label="Merge method",
                                choices=list(REGISTERED_MERGE_METHODS.keys()),
                                value=list(REGISTERED_MERGE_METHODS.keys())[0],
                                interactive=True,
                            )
                        with gr.Column():
                            models = []

                            def add_model(model: str):
                                models.append({"model": model, "parameters": {}})
                                return ""  # blank out the search box

                            with gr.Row():
                                gui_search = HuggingfaceHubSearch(
                                    label="Search for a model", interactive=True
                                )
                                btn_gui_search = gr.Button("Add model")
                                btn_gui_search.click(
                                    add_model, inputs=[gui_search], outputs=[gui_search]
                                )

                            @gr.render(
                                inputs=[gui_search, merge_method],
                                triggers=[btn_gui_search.click, merge_method.change],
                            )
                            def render_models(model: str, merge_method: str):
                                merge_method: MergeMethod = REGISTERED_MERGE_METHODS[
                                    merge_method
                                ]
                                tensor_params = merge_method.tensor_parameters()
                                for model in models:

                                    def remove_model():
                                        models.remove(model)

                                    with gr.Row(variant="panel"):
                                        with gr.Column():
                                            HuggingfaceHubSearch(
                                                label="Model", value=model["model"]
                                            )
                                            with gr.Accordion("Parameters", open=False):
                                                for param in tensor_params:

                                                    def update_params(param_value: str):
                                                        model["parameters"][
                                                            param.name
                                                        ] = param_value

                                                    box = gr.Textbox(
                                                        label=param.name,
                                                        value=param.default_value,
                                                        interactive=True,
                                                    )
                                                    box.change(
                                                        update_params, inputs=[box]
                                                    )
                                        btn = gr.Button("Remove model")
                                        btn.click(remove_model)

                    with gr.Column():
                        btn_gui = gr.Button("Merge!")
                        Log(log_file=tmp.name, dark=True, label="Logs", container=True)
        gr.LoginButton()

        btn_yaml.click(do_merge, inputs=[yaml_config, state, output_model_path])
        if not PROD:
            btn_gui.click(lambda: print(models))

    tmp.close()

    gui.queue(default_concurrency_limit=1).launch(share=share)


@click.command("mergekitty-gui")
@click.option(
    "--share", is_flag=True, required=False, default=False, help="Share the GUI"
)
def cli_main(share: bool):
    main(share)


if __name__ == "__main__":
    cli_main()
