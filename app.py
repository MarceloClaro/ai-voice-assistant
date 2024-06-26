import streamlit as st
import speech_recognition as sr
import openai
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings

# Configurar a chave da API OpenAI
openai.api_key = os.getenv("sk-proj-VqQdiflImI1O4LIBn8OBT3BlbkFJR8nstd556kCDmZ66ztmZ")

# Função para transcrever áudio
def transcrever_audio(nome_arquivo):
    recognizer = sr.Recognizer()
    with sr.AudioFile(nome_arquivo) as source:
        audio_data = recognizer.record(source)
        texto = recognizer.recognize_google(audio_data, language='pt-BR')
    return texto

# Função para gerar resposta
def gerar_resposta(pergunta):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=pergunta,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Função principal para processar áudio
def process_audio(audio_file_path):
    st.info("Transcrevendo o áudio...")
    texto_transcrito = transcrever_audio(audio_file_path)
    st.session_state.conversas.append(f"Usuário: {texto_transcrito}")
    st.write(f"Transcrição: {texto_transcrito}")

    st.info("Gerando a resposta do agente...")
    resposta = gerar_resposta(texto_transcrito)
    st.session_state.conversas.append(f"Agente: {resposta}")
    st.write(f"Resposta do Agente: {resposta}")

# Configurações do WebRTC
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    media_stream_constraints={
        "audio": True,
        "video": False,
    },
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# Função para inicializar o WebRTC e gravar áudio
def webrtc_audio_recorder():
    webrtc_ctx = webrtc_streamer(
        key="audio_recorder",
        mode=WebRtcMode.SENDRECV,
        client_settings=WEBRTC_CLIENT_SETTINGS,
    )

    if webrtc_ctx.state.playing:
        if st.button("Parar Gravação"):
            webrtc_ctx.stop()
            audio_data = webrtc_ctx.audio_receiver.get_audio_data()
            if audio_data:
                with open("audio.wav", "wb") as f:
                    f.write(audio_data)
                process_audio("audio.wav")

# Interface do Streamlit
st.title("Agente 4 - Gravador, Transcritor e Respondedor de Áudio")

if "conversas" not in st.session_state:
    st.session_state.conversas = []

webrtc_audio_recorder()

st.markdown("### Histórico de Conversas")
for conversa in st.session_state.conversas:
    st.write(conversa)
