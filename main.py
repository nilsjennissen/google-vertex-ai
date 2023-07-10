''' This is the main application file for the CODER application. It contains the main functions and the support
    functions for the application. '''

#%% ----------------------------- IMPORTS  -----------------------------------
# import libraries
import streamlit as st
import pandas as pd
import numpy as np
import credentials
import os
import openai
from gtts import gTTS
import io
from IPython.display import Audio
import time
#import en_core_web_sm
import spacy_streamlit
from pydub import AudioSegment
import matplotlib.pyplot as plt
from audio_recorder_streamlit import audio_recorder
from IPython.display import Audio
from langchain.callbacks import get_openai_callback
# Import the Speech-to-Text client library
from google.cloud import speech

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech



#%% ----------------------------- LANGCHAIN FUNCTIONS -----------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
OPENAI_API_KEY = credentials.OPENAI_API_KEY


# %% ----------------------------- AUDIO RECORDING STREAMLIT -----------------------------------
def rec_streamlit():
    """Record audio and return the audio bytes"""
    audio_bytes = audio_recorder(energy_threshold=(-1.0, 1.0), pause_threshold=6.0, text="",
                                 recording_color="#FF0000", neutral_color="#49DE49", icon_name="microphone", icon_size="3x")

    return audio_bytes


#%% ----------------------------- OPEN AI FUNCTIONS  -----------------------------------
def get_transcript_whisper(file_path):
    '''Get the transcript of the audio file'''
    openai.api_key = OPENAI_API_KEY
    with open(file_path, "rb") as file:
        transcription = openai.Audio.transcribe("whisper-1", file, response_format="json")
    transcribed_text = transcription["text"]

    return transcribed_text


#%% ----------------------------- GOOGLE CLOUD SPEECH V1  -----------------------------------
def get_transcript_google(file_path):
    """Transcribe the given audio file."""
    client = speech.SpeechClient()

    with open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,
            language_code="en-US",
            model="default",
            audio_channel_count=1,
            enable_word_confidence=True,
            enable_word_time_offsets=True,
    )

    response = client.recognize(config=config, audio=audio)

    return response



def quickstart_v2(
    project_id: str,
    audio_file: str,
) -> cloud_speech.RecognizeResponse:
    """Transcribe an audio file."""
    # Instantiates a client
    client = SpeechClient()

    # Reads a file as bytes
    with open(audio_file, "rb") as f:
        content = f.read()

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config={}, language_codes=["en-US"], model="latest_long"
    )

    request = cloud_speech.RecognizeRequest(
        recognizer=f"projects/{project_id}/locations/global/recognizers/_",
        config=config,
        content=content,
    )

    # Transcribes the audio into text
    response = client.recognize(request=request)

    for result in response.results:
        print(f"Transcript: {result.alternatives[0].transcript}")

    return response


#%% ----------------------------- SPEAK FUNCTIONS -----------------------------------
def speak_answer(answer, tts_enabled):
    if not tts_enabled:
        return

    tts = gTTS(text=answer, lang="en")
    with io.BytesIO() as f:
        tts.write_to_fp(f)
        f.seek(0)
        audio = Audio(f.read(), autoplay=True)
        st.write(audio)


#%% ----------------------------- MAIN APPLICATION FUNCTIONS -----------------------------------
def home():
    # ------------------ SETTINGS ------------------
    st.set_page_config(page_title="Home", layout="wide")
    st.markdown("""<style>.reportview-container .main .block-container {max-width: 95%;}</style>""",
                unsafe_allow_html=True)

    # ------------------ HOME PAGE ------------------
    st.title("MAIN FILE MULTIAGENT 🎙️📖🥷")
    st.write("""Use the power of LLMs with LangChain and OpenAI to scan through your documents. Find information 
    and insight's with lightning speed. 🚀 Create new content with the support of state of the art language models and 
    and voice command your way through your documents. 🎙️""")
    st.write("Let's start interacting with GPT-4!")

    # ------------------ SIDE BAR SETTINGS ------------------
    st.sidebar.subheader("Settings:")
    tts_enabled = st.sidebar.checkbox("Enable Text-to-Speech", value=False)
    ner_enabled = st.sidebar.checkbox("Enable NER in Response", value=False)


# Run home function
if __name__ == "__main__":
    home()
