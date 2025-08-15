import gradio as gr
import torch
from model import load_model, predict_next_word
import os

# Load your trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vocab = load_model(device)

# Prediction function
def predict(text_input, top_k=5):
    try:
        predictions = predict_next_word(model, vocab, text_input, top_k=top_k, device=device)
        return ", ".join(predictions)
    except Exception as e:
        return f"Error: {str(e)}"

# Custom CSS
custom_css = """
body { background-color: #1e1e2f; color: #ffffff; font-family: 'Segoe UI', sans-serif; }
.gr-button { background-color: #ff5e5e !important; color: white !important; border-radius: 8px; font-weight: bold; }
.gr-slider { color: #ff5e5e; }
.gr-textbox { background-color: #2e2e3e; color: #ffffff; border-radius: 6px; }
.gr-markdown { color: #ffffff; }
"""

# Gradio Blocks UI
with gr.Blocks(css=custom_css, title="🔎 Sherlock Holmes Next Word Predictor") as demo:

    # Header with logo and title
    with gr.Row():
        gr.Image("sherlock_logo.png", elem_id="logo", width=100)  # Place a logo in the folder
        gr.Markdown("<h1 style='margin-left:15px;'>Sherlock Holmes Next Word Prediction</h1>")

    gr.Markdown("""
    Enter a snippet from Sherlock Holmes stories, and the model will predict the next word!  
    Adjust the number of predictions with the slider.
    """)

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Input Text", placeholder="Type a Sherlock Holmes sentence...", lines=4)
            top_k = gr.Slider(1, 10, value=5, step=1, label="Number of predictions (top-k)")
            predict_btn = gr.Button("Predict Next Word")
        with gr.Column(scale=1):
            output = gr.Textbox(label="Predicted Next Words", placeholder="Predictions will appear here...")

    # Button click action
    predict_btn.click(fn=predict, inputs=[text_input, top_k], outputs=output)

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
