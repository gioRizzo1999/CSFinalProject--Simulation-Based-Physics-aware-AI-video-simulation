# Gradio app for minimalistic front-end, starting from documentation template
import gradio as gr
import run_pipeline

# User can add prompt to describe video scene and then triggere generation pipeline
with gr.Blocks(css="""
#result_video video {
    max-height: 260px;
    width: auto;
}
#result_video {
    max-width: 340px;
    margin: 0 auto;
}
""") as demo:
    user_input = gr.Textbox(label="Scene description")
    btn = gr.Button("Generate Video")
    media_out = gr.Video(elem_id="output_video")
    btn.click(
        fn=run_pipeline.run_pipeline, 
        inputs=user_input, 
        outputs=media_out)
demo.launch()


