# https://www.gradio.app/guides/using-hugging-face-integrations
import gradio as gr

from transformers import pipeline

TASK = "text-generation"
MODEL = "openlm-research/open_llama_3b_v2"
# MODEL = "Helsinki-NLP/opus-mt-en-es"

pipe = pipeline(TASK, model=MODEL, max_length=200)

def predict(text):
  out = pipe(text)[0]['generated_text']
  return out
  # return pipe(text)[0]["translation_text"]

demo = gr.Interface(
  fn=predict,
  inputs='text',
  outputs='text',
)

demo.launch()