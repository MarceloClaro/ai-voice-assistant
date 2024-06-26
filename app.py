#groq_api_key = os.getenv('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')

import streamlit as st
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import chainlit as cl
from dotenv import load_dotenv
import os

# Carregando variáveis de ambiente do arquivo .env
load_dotenv() 

# Inicializando a API Key do modelo
groq_api_key = os.getenv('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')

# Substituir o inicializador de ChatGroq por uma função de log ou um LLM disponível.
# Por exemplo, substitua por um LLM local ou qualquer outro modelo compatível com langchain.
# Aqui vamos usar um modelo placeholder.
from langchain.llms import OpenAI
llm_groq = OpenAI(temperature=0.2, openai_api_key=groq_api_key)

@cl.on_chat_start
async def on_chat_start():
    files = None  # Inicializar variável para armazenar arquivos enviados

    # Aguardar o usuário enviar arquivos
    while files is None:
        files = await cl.AskFileMessage(
            content="Por favor, envie um ou mais arquivos PDF para começar!",
            accept=["application/pdf"],
            max_size_mb=100,  # Limitar opcionalmente o tamanho do arquivo
            max_files=10,
            timeout=180,  # Definir um tempo limite para resposta do usuário
        ).send()

    # Processar cada arquivo enviado
    texts = []
    metadatas = []
    for file in files:
        print(file)  # Imprimir o objeto arquivo para depuração

        # Ler o arquivo PDF
        pdf = PyPDF2.PdfReader(file.path)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        # Dividir o texto em partes
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        # Criar metadados para cada parte
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)

    # Criar uma loja de vetores Chroma
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    # Inicializar o histórico de mensagens para conversa
    message_history = ChatMessageHistory()

    # Memória para contexto conversacional
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Criar uma cadeia que usa a loja de vetores Chroma
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_groq,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Enviar uma imagem com o número de arquivos processados
    elements = [
        cl.Image(name="image", display="inline", path="pic.jpg")
    ]
    # Informar ao usuário que o processamento terminou. Agora você pode conversar.
    msg = cl.Message(content=f"Processamento de {len(files)} arquivos concluído. Você já pode fazer perguntas!", elements=elements)
    await msg.send()

    # Armazenar a cadeia na sessão do usuário
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    # Recuperar a cadeia da sessão do usuário
    chain = cl.user_session.get("chain")
    # Callbacks acontecem de forma assíncrona/paralela
    cb = cl.AsyncLangchainCallbackHandler()

    # Chamar a cadeia com o conteúdo da mensagem do usuário
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]

    text_elements = []  # Inicializar lista para armazenar elementos de texto

    # Processar documentos de origem se disponíveis
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Criar o elemento de texto referenciado na mensagem
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        # Adicionar referências de fonte à resposta
        if source_names:
            answer += f"\nFontes: {', '.join(source_names)}"
        else:
            answer += "\nNenhuma fonte encontrada"
    
    # Retornar resultados
    await cl.Message(content=answer, elements=text_elements).send()
