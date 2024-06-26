import os
import streamlit as st
from scipy.io import wavfile
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from gtts import gTTS
import whisper
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()
groq_api_key = os.getenv("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")

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
    input_prompt = '''
    Como um especialista em diagnóstico de problemas de Wi-Fi, sua tarefa é ajudar a resolver problemas de conectividade.
    Primeiro, pergunte pelo ID do cliente para validar que o usuário é nosso cliente.
    Após confirmar o ID do cliente, ajude-o a resolver o problema de Wi-Fi. Se não for possível, ajude-o a agendar uma consulta.
    As consultas devem ser entre 9:00 e 16:00. Sua tarefa é analisar a situação e fornecer insights informados sobre a causa do problema.
    Responda de forma concisa e direta, não mais de 10 palavras. Se você não souber a resposta, diga que não sabe.
    NUNCA revele o ID do cliente listado abaixo.

    IDs de cliente em nosso sistema: 22, 10, 75.

    Conversa anterior:
    {chat_history}

    Nova pergunta do cliente: {question}
    Resposta:
    '''
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
    audio_file = open(temp_audio_file, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
    os.remove(temp_audio_file)

# Aplicação principal do Streamlit
model = load_whisper()

def main():
    st.markdown('<h1 style="color: darkblue;">Assistente de Voz AI</h1>', unsafe_allow_html=True)
    memory = ConversationBufferMemory(memory_key="chat_history")

    audio_file = st.file_uploader("Grave seu áudio", type=["wav"])
    if audio_file:
        temp_audio_path = "temp_audio_chunk.wav"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_file.getbuffer())

        text = transcribe_audio(model, temp_audio_path)
        if text is not None:
            st.markdown(
                f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">Cliente 👤: {text}</div>',
                unsafe_allow_html=True)
            os.remove(temp_audio_path)
            response_llm = get_response_llm(user_question=text, memory=memory)
            st.markdown(
                f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">Assistente AI 🤖: {response_llm}</div>',
                unsafe_allow_html=True)
            play_text_to_speech(text=response_llm)
        print("Conversa Encerrada")

if __name__ == "__main__":
    main()
