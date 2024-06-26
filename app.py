import streamlit as st
import numpy as np
import soundfile as sf
import os
import openai
from gtts import gTTS
from io import BytesIO
import time
import queue
import sounddevice as sd
import whisper
import ffmpeg
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Ajuste de chave da API
openai.api_key = os.getenv('sk-proj-VqQdiflImI1O4LIBn8OBT3BlbkFJR8nstd556kCDmZ66ztmZ')

# Funções para gravação de áudio e processamento
def record_audio(duration, fs=44100, channels=1):
    q = queue.Queue()

    def callback(indata, frames, time, status):
        q.put(indata.copy())

    with sd.InputStream(samplerate=fs, channels=channels, callback=callback):
        st.text("Gravando...")
        time.sleep(duration)
    
    data = []
    while not q.empty():
        data.append(q.get())
    return np.concatenate(data)

def save_audio(file_path, data, fs):
    sf.write(file_path, data, fs)

def transcribe_audio(file_path):
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result['text']

def play_audio(file_path):
    data, fs = sf.read(file_path, dtype='float32')
    sd.play(data, fs)
    sd.wait()

# Função principal do Streamlit
def main():
    st.title("Agente de Áudio com Memória")

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    duration = st.slider("Duração da gravação (segundos):", 1, 10, 5)
    if st.button("Gravar"):
        audio_data = record_audio(duration)
        file_path = "recorded_audio.wav"
        save_audio(file_path, audio_data, 44100)

        st.session_state.conversation.append({"role": "user", "text": "Áudio gravado"})
        st.audio(file_path, format='audio/wav')

        transcription = transcribe_audio(file_path)
        st.session_state.conversation.append({"role": "user", "text": transcription})
        st.text("Transcrição: " + transcription)

        llm = OpenAI(temperature=0.5)
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="{history}\nUser: {input}\nAssistant:"
        )
        chain = LLMChain(llm=llm, prompt=prompt)

        history = "\n".join([f"{x['role'].capitalize()}: {x['text']}" for x in st.session_state.conversation])
        response = chain.run(history=history, input=transcription)

        st.session_state.conversation.append({"role": "assistant", "text": response})
        st.text("Resposta do Assistente: " + response)

        tts = gTTS(response, lang='pt')
        audio_response_path = "response_audio.wav"
        tts.save(audio_response_path)
        st.audio(audio_response_path, format='audio/wav')

        if st.button("Reproduzir Resposta"):
            play_audio(audio_response_path)

    if st.button("Mostrar Conversa"):
        for msg in st.session_state.conversation:
            st.text(f"{msg['role'].capitalize()}: {msg['text']}")

if __name__ == "__main__":
    main()
