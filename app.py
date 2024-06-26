import streamlit as st
import PyPDF2
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Chave da API Groq
groq_api_key = os.getenv('GROQ_API_KEY')

# Inicializar o chat Groq com a chave API fornecida
llm_groq = ChatGroq(
    groq_api_key=groq_api_key, model_name="llama3-70b-8192",
    temperature=0.2
)

def process_files(files):
    texts = []
    metadatas = []
    for file in files:
        # Ler o arquivo PDF
        pdf = PyPDF2.PdfReader(file)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
            
        # Dividir o texto em pedaços
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=50)
        file_texts = text_splitter.split_text(pdf_text)
        texts.extend(file_texts)

        # Criar metadados para cada pedaço
        file_metadatas = [{"source": f"{i}-{file.name}"} for i in range(len(file_texts))]
        metadatas.extend(file_metadatas)

    # Criar um repositório de vetores Chroma
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=metadatas)
    
    return docsearch

def main():
    st.title("Conversational Retrieval with Groq and Streamlit")
    
    # Upload de arquivos PDF
    uploaded_files = st.file_uploader("Upload one or more PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) uploaded.")
        docsearch = process_files(uploaded_files)
        
        # Inicializar histórico de mensagens
        message_history = ChatMessageHistory()
        
        # Memória para contexto conversacional
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            chat_memory=message_history,
            return_messages=True,
        )
        
        # Criar uma cadeia que usa o repositório de vetores Chroma
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm_groq,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            memory=memory,
            return_source_documents=True,
        )
        
        if 'chain' not in st.session_state:
            st.session_state.chain = chain
        
        st.success("Files processed. You can now ask questions!")
        
        user_question = st.text_input("Ask a question about the uploaded documents:")
        
        if user_question:
            response = st.session_state.chain({"query": user_question})
            answer = response["answer"]
            source_documents = response["source_documents"]
            
            st.write(f"**Answer:** {answer}")
            
            if source_documents:
                st.write("**Sources:**")
                for idx, doc in enumerate(source_documents):
                    st.write(f"{idx + 1}. {doc.page_content}")

if __name__ == "__main__":
    main()
