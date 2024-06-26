import streamlit as st
import pyaudio
import wave
import speech_recognition as sr
import openai
import base64
import os

# Configurar a chave da API OpenAI
openai.api_key = os.getenv("sk-proj-VqQdiflImI1O4LIBn8OBT3BlbkFJR8nstd556kCDmZ66ztmZ")

# Funções para gravação e transcrição de áudio
def gravar_audio(duracao=5, nome_arquivo="gravacao.wav"):
    formato = pyaudio.paInt16
    canais = 1
    taxa = 44100
    chunk = 1024

    audio = pyaudio.PyAudio()
    stream = audio.open(format=formato, channels=canais, rate=taxa, input=True, frames_per_buffer=chunk)
    st.info("Gravando...")

    frames = []
    for _ in range(0, int(taxa / chunk * duracao)):
        data = stream.read(chunk)
        frames.append(data)

    st.info("Gravação concluída")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    with wave.open(nome_arquivo, 'wb') as wf:
        wf.setnchannels(canais)
        wf.setsampwidth(audio.get_sample_size(formato))
        wf.setframerate(taxa)
        wf.writeframes(b''.join(frames))

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
    audio_html = f"""
        <audio controls autoplay>
            <source src="data:audio/wav;base64,{base64.b64encode(open(nome_arquivo, "rb").read()).decode()}" type="audio/wav">
            Seu navegador não suporta o elemento de áudio.
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

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
