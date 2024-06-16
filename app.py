import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain 
from streamlit_chat import message



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function=len
        )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    converation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return converation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, mess in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(mess.content, is_user = True,key=f'user_message_{i}')
        else:
            message(mess.content,key=f'bot_message_{i}')


def main():
    import os
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv(dotenv_path="D:\Courses and materials\TLP Plaksha\Term 4\Test code\key.env")

    # Now you can access your API key as an environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    link='https://en.wikipedia.org/wiki/Culture_of_Bengal'
    st.set_page_config(page_title="Know Bengal", page_icon=":performing_arts:")
   
    st.image("Art-and-Culture-of-West-Bengal-1068x712.jpg", use_column_width=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    
    st.header(" Know How's of Bengal:link:")
    user_question = st.text_input("Ask more about Bengal :")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload any PDFs here and click on process", accept_multiple_files = True)
        if st.button("Process"):
            with st.spinner("Processing"):

                # get pdf text
                raw_text = get_pdf_text(pdf_docs)


                #get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

    message("Hello! Ask anything about Bengal in context of Uploaded Documents.")
    message("Okay!", is_user=True)

if __name__=="__main__":
    main()