import os
import streamlit as st
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

st.title("News Research Tool")
st.sidebar.title("News Article URLs")


load_dotenv()

urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Search")
main_placeholder = st.empty()

llm = OpenAI(temperature=0.9, max_tokens=500)
embeddings = OpenAIEmbeddings()

if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,chunk_overlap=200
    )
    docs = text_splitter.split_documents(data)

    vectorstore_openai = FAISS.from_documents(docs, embeddings)

    vectorstore_openai.save_local("vector_index")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists("vector_index"):
        vectorstore = FAISS.load_local("vector_index", embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        st.header("Results")
        st.write(result["answer"])
