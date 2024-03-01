import os
import openai
import pandas as pd

from datetime import datetime

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFium2Loader,
    UnstructuredExcelLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_community.llms import LlamaCpp
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
vector_store_path = "./faiss_index"
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_text_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        keep_separator=False,
        separators=["\n\n", "\n", ""],
    )
    chunks = text_splitter.split_documents(docs)
    return chunks


def process_documents(file_path):
    loader = UnstructuredExcelLoader(file_path, mode="elements")
    documents = loader.load()
    # chunks = get_text_chunks(documents)
    return documents


def ingest_files(file_path):
    splitted_documents = process_documents(file_path)
    vectordb = FAISS.from_documents(splitted_documents, embedding=embeddings)
    vectordb.save_local(vector_store_path)
    return splitted_documents


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_retriever():
    vectorstore = FAISS.load_local(vector_store_path, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    return retriever


def get_prompt():
    template = """You are a helpful AI assistant who will act as an Analyst. Analyse the given set of requirements and prepare a brief presentation for Request for Proposal.
    Question: {question}
    Context: {context}
    Answer:
    """

    prompt_template = PromptTemplate(
        template=template,
        input_variables=[
            "context",
            "question",
        ],
    )

    return prompt_template


def llm_chain_openai():
    llm = ChatOpenAI(
        model="gpt-4-0125-preview",
        openai_api_key=os.environ["OPENAI_API_KEY"],
        temperature=0.5,
    )

    retriever = get_retriever()
    prompt = get_prompt()

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


if __name__ == "__main__":
    # pdf_path = "./req.xlsx"
    # splitted_text = ingest_files(pdf_path)

    chain = llm_chain_openai()

    # What is the purpose of the document?
    result = chain.invoke(
        "Can you create consolidate workflow of the Investor approval process?",
    )

    print(result)
