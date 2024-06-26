import streamlit as st
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import os
import openai
from scipy.io.wavfile import write

# Configurar a chave da API OpenAI
openai.api_key = os.getenv("sk-proj-VqQdiflImI1O4LIBn8OBT3BlbkFJR8nstd556kCDmZ66ztmZ")

# Funções para gravação e transcrição de áudio
def gravar_audio(duracao=5, nome_arquivo="gravacao.wav"):
    fs = 44100  # Sample rate
    st.info("Gravando...")
    myrecording = sd.rec(int(duracao * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()  # Espera até a gravação terminar
    write(nome_arquivo, fs, myrecording)  # Salva o arquivo como WAV
    st.info("Gravação concluída")
    return nome_arquivo

def transcrever_audio(nome_arquivo):
    recognizer = sr.Recognizer()
    with sr.AudioFile(nome_arquivo) as source:
        audio_data = recognizer.record(source)
        texto = recognizer.recognize_google(audio_data, language='pt-BR')
    return texto

def gerar_resposta(pergunta):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=pergunta,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def tocar_audio(nome_arquivo):
    audio_placeholder = st.empty()
    audio_bytes = open(nome_arquivo, 'rb').read()
    audio_html = f"""
        <audio controls autoplay>
            <source src="data:audio/wav;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/wav">
            Seu navegador não suporta o elemento de áudio.
        </audio>
    """
    audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

st.title("Agente 4 - Gravador, Transcritor e Respondedor de Áudio")

if "conversas" not in st.session_state:
    st.session_state.conversas = []

if st.button("Gravar Áudio"):
    nome_arquivo = gravar_audio()
    texto_transcrito = transcrever_audio(nome_arquivo)
    st.session_state.conversas.append(f"Usuário: {texto_transcrito}")
    st.write(f"Transcrição: {texto_transcrito}")

    resposta = gerar_resposta(texto_transcrito)
    st.session_state.conversas.append(f"Agente: {resposta}")
    st.write(f"Resposta do Agente: {resposta}")
    tocar_audio(nome_arquivo)

st.markdown("### Histórico de Conversas")
for conversa in st.session_state.conversas:
    st.write(conversa)
