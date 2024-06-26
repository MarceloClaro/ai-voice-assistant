import os
import streamlit as st
import numpy as np
import openai
import tempfile
from gtts import gTTS
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import queue
import wave

# Configurar a chave da API OpenAI
openai.api_key = os.getenv("sk-proj-VqQdiflImI1O4LIBn8OBT3BlbkFJR8nstd556kCDmZ66ztmZ")

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recording = queue.Queue()
        self.running = False

    def recv(self, frame):
        if self.running:
            self.recording.put(frame.to_ndarray().flatten())
        return frame

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

def record_audio(duration):
    ctx = webrtc_streamer(key="example", audio_processor_factory=AudioProcessor, media_stream_constraints={"audio": True})

    if st.button("Iniciar Gravação"):
        ctx.audio_processor.start()
        st.write("Gravando...")

        st.sleep(duration)

        ctx.audio_processor.stop()
        st.write("Gravação finalizada.")

        # Process the recorded audio
        frames = []
        while not ctx.audio_processor.recording.empty():
            frames.append(ctx.audio_processor.recording.get())

        audio_data = np.concatenate(frames, axis=0).astype(np.int16)
        return audio_data

    return None

def save_audio_file(audio_data, filename="output.wav"):
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)  # mono
        wf.setsampwidth(2)  # 16 bits per sample
        wf.setframerate(44100)  # sample rate
        wf.writeframes(audio_data.tobytes())

def transcribe_audio(filename):
    with open(filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript

def generate_response(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def text_to_speech(text, lang="pt"):
    tts = gTTS(text=text, lang=lang)
    tts_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tts_file.name)
    return tts_file.name

def play_audio(file_path):
    audio_data = open(file_path, "rb").read()
    st.audio(audio_data, format="audio/mp3")

# Interface do Streamlit
st.title("Agente 4 - Grava, Transcreve, Responde e Reproduz Áudio")

duration = st.number_input("Duração da gravação (segundos):", min_value=1, max_value=60, value=5)

audio_data = record_audio(duration)

if audio_data is not None:
    save_audio_file(audio_data)

    transcript = transcribe_audio("output.wav")
    st.write("Transcrição:", transcript)

    response = generate_response(transcript)
    st.write("Resposta:", response)

    tts_file = text_to_speech(response)
    play_audio(tts_file)
