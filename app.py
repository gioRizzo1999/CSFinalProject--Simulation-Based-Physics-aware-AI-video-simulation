# Gradio app for minimalistic front-end, starting from documentation template
import gradio as gr
import run_pipeline

# User can add prompt to describe video scene and then triggere generation pipeline
with gr.Blocks(css="""
#output_video {
    width: 740px;
    max-width: 740px;
    margin: 0 auto;
}
#output_video video {
    width: 100%;
    max-height: 760px;
    object-fit: contain;
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


