import os
import streamlit as st
import pickle
import time
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain_community.vectorstores import FAISS       
import faiss
import numpy as np

st.title("News Research Tool 📈")
st.sidebar.title("News Articles URLs")

urls=[]
file="vector_index.pkl"

for i in range(3):
    url=st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URLs")
main_placefolder=st.empty()
if process_url_clicked:

    loader=UnstructuredURLLoader(urls=urls)

    main_placefolder.text("data loading started ✅")
    data=loader.load()

    text_splitter=RecursiveCharacterTextSplitter(

        chunk_size=100,
        chunk_overlap=50,
        separators=["\n\n","\n","."]
    )
    main_placefolder.text("Chunk splitting started ➡️")
    documents=text_splitter.split_documents(data)

    # st.write(documents)

    main_placefolder.text("embedding ✅")
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = FAISS.from_documents(documents=documents,embedding=embeddings)
    with open(file, "wb") as f:
         pickle.dump(vector_store, f)



query=main_placefolder.text_input("Question :")

if query:

        if os.path.exists(file):

            with open(file,"rb") as f:

                vector_st=pickle.load(f)

                llm = OllamaLLM(model="llama3.1")
                chain=RetrievalQAWithSourcesChain.from_llm(
                    llm=llm,
                    retriever=vector_st.as_retriever()
                )
                result=chain({"question":query},return_only_outputs=True)

                st.header("Answer")
                st.subheader(result["answer"])

                



    


