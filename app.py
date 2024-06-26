import os
import wave
import pyaudio
import streamlit as st
import numpy as np
from scipy.io import wavfile
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from gtts import gTTS
import pygame
import whisper
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

def is_silence(data, max_amplitude_threshold=3000):
    """Verificar se os dados de áudio contêm silêncio."""
    max_amplitude = np.max(np.abs(data))
    return max_amplitude <= max_amplitude_threshold

def record_audio_chunk(audio, stream, chunk_length=5):
    print("Gravando...")
    frames = []
    num_chunks = int(16000 / 1024 * chunk_length)

    for _ in range(num_chunks):
        data = stream.read(1024)
        frames.append(data)

    temp_file_path = 'temp_audio_chunk.wav'
    print("Escrevendo...")
    with wave.open(temp_file_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))

    try:
        samplerate, data = wavfile.read(temp_file_path)
        if is_silence(data):
            os.remove(temp_file_path)
            return True
        else:
            return False
    except Exception as e:
        print(f"Erro ao ler o arquivo de áudio: {e}")

def load_whisper():
    model = whisper.load_model("base")
    return model

def transcribe_audio(model, file_path):
    print("Transcrevendo...")
    if os.path.isfile(file_path):
        results = model.transcribe(file_path)
        return results['text']
    else:
        return None

def load_prompt():
    input_prompt = """
    Como um consultor especializado em diagnosticar problemas de Wi-Fi, sua experiência é fundamental para solucionar e
    resolver problemas de conectividade. Primeiro, peça o ID do cliente para validar se o usuário é nosso cliente. 
    Após confirmar o ID do cliente, ajude-o a resolver o problema de Wi-Fi, se não for possível, ajude-o a marcar uma 
    consulta. As consultas devem ser entre 9:00 e 16:00. Sua tarefa é analisar
    a situação e fornecer insights informados sobre a causa raiz da interrupção do Wi-Fi. Forneça respostas concisas e curtas
    com no máximo 10 palavras, e não converse consigo mesmo!. Se você não souber a resposta,
    diga apenas que não sabe, não tente inventar uma resposta. NUNCA diga o ID do cliente listado abaixo.

    IDs de cliente em nossos dados: 22, 10, 75.

    Conversa anterior:
    {chat_history}

    Nova pergunta do humano: {question}
    Resposta:
    """
    return input_prompt

def load_llm():
    chat_groq = ChatGroq(temperature=0, model_name="llama3-8b-8192",
                         groq_api_key=groq_api_key)
    return chat_groq

def get_response_llm(user_question, memory):
    input_prompt = load_prompt()
    chat_groq = load_llm()
    prompt = PromptTemplate.from_template(input_prompt)
    chain = LLMChain(
        llm=chat_groq,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    response = chain.invoke({"question": user_question})
    return response['text']

def play_text_to_speech(text, language='pt', slow=False):
    tts = gTTS(text=text, lang=language, slow=slow)
    temp_audio_file = "temp_audio.mp3"
    tts.save(temp_audio_file)
    pygame.mixer.init()
    pygame.mixer.music.load(temp_audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.stop()
    pygame.mixer.quit()
    os.remove(temp_audio_file)

# Aplicação principal Streamlit
chunk_file = 'temp_audio_chunk.wav'
model = load_whisper()

def main():
    st.markdown('<h1 style="color: darkblue;">Assistente de Voz AI</h1>', unsafe_allow_html=True)
    memory = ConversationBufferMemory(memory_key="chat_history")

    if st.button("Iniciar Gravação"):
        while True:
            audio = pyaudio.PyAudio()
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
            record_audio_chunk(audio, stream)
            text = transcribe_audio(model, chunk_file)

            if text is not None:
                st.markdown(
                    f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">Cliente 👤: {text}</div>',
                    unsafe_allow_html=True)
                os.remove(chunk_file)
                response_llm = get_response_llm(user_question=text, memory=memory)
                st.markdown(
                    f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">Assistente AI 🤖: {response_llm}</div>',
                    unsafe_allow_html=True)
                play_text_to_speech(text=response_llm)
            else:
                stream.stop_stream()
                stream.close()
                audio.terminate()
                break
        print("Fim da Conversa")

if __name__ == "__main__":
    main()
