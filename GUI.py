#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install gradio')
get_ipython().system('pip install openai-whisper')
get_ipython().system('pip install sentencepiece')


# In[ ]:


import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import librosa
import whisper
import gradio as gr

def load_wav2vec_model(language):
    """Load the appropriate Wav2Vec2 model for English and Hindi."""
    model_mapping = {
        "english": "facebook/wav2vec2-large-960h",
        "hindi": "theainerd/Wav2Vec2-large-xlsr-hindi"
    }

    if language not in model_mapping:
        raise ValueError("Unsupported language for Wav2Vec2. Use Whisper for Tamil, Telugu, Malayalam.")

    model_name = model_mapping[language]
    print(f"Loading Wav2Vec2 model: {model_name}")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model

def transcribe_wav2vec(audio_path, language):
    """Transcribe speech using Wav2Vec2 model."""
    processor, model = load_wav2vec_model(language)

    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    input_values = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

def transcribe_whisper(audio_path):
    """Transcribe speech using Whisper model for Tamil, Telugu, Malayalam."""
    model = whisper.load_model("medium")
    result = model.transcribe(audio_path)
    return result["text"]

def transcribe(audio, language):
    """Gradio function to handle audio input and transcribe."""
    if audio is None:
        return "No audio file provided."

    print(f"Received audio file: {audio}")

    if language in ["english", "hindi"]:
        transcription = transcribe_wav2vec(audio, language)
    elif language in ["tamil", "telugu", "malayalam"]:
        transcription = transcribe_whisper(audio)
    else:
        return "Unsupported language."

    return f"Transcription: {transcription}"

iface = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(type="filepath"),
        gr.Dropdown(["english", "hindi", "tamil", "malayalam", "telugu"], label="Select Language")
    ],
    outputs="text",
    title="Multilingual Speech-to-Text",
    description="Upload an audio file, select a language, and get the transcribed text."
)

if __name__ == "__main__":
    print("Launching Gradio interface...")
    iface.launch(debug=True)

