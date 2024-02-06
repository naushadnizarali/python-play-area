import datetime
import os
from sentence_extractor import create_vector_db
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain,
    StuffDocumentsChain,
    QAGenerationChain,
)
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from sentence_extractor import process_files, create_vector_db
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# INSTRUCTOR_EMBEDDINGS = HuggingFaceInstructEmbeddings(
#     model_name="hkunlp/instructor-large"
# )

EMBEDDINGS = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

VECTOR_STORE_PATH = "faiss_index_individual_2"

CHAT_MEMORY = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# CHAT_MEMORY.save_context(
#     {"role": "system", "content": "You are a helpful assistant."},
#     {
#         "role": "user",
#         "content": "I have a document about description and specification of a software project.",
#     },
# )
TEMP = 0.3

LLM_QUESTION = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-3.5-turbo-0125",
    temperature=TEMP,
    max_tokens=4096,
)

LLM_CHAT = ChatOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    model="ft:gpt-3.5-turbo-1106:mobilelive-inc:irap-model:8mTN1w61",
    temperature=TEMP,
    max_tokens=4096,
)


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
    loader = PyPDFium2Loader(file_path)
    documents = loader.load()
    chunks = get_text_chunks(documents)
    return chunks


def ingest_files(file_path: os.PathLike) -> None:
    splitted_documents = process_documents(file_path)
    vectordb = FAISS.from_documents(splitted_documents, embedding=EMBEDDINGS)
    vectordb.save_local(VECTOR_STORE_PATH)


def get_retriever(db_path):
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(db_path, EMBEDDINGS)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    return retriever


def document_chain():
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    document_variable_name = "context"
    llm = OpenAI()

    prompt = PromptTemplate.from_template("Summarize this content: {context}")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )

    return chain


def question_chain():
    # Create OpenAI LLM model

    _template = """Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question, in its original language.
    Follow Up Input: {question}
    Chat History:
    {chat_history}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    chain = LLMChain(
        llm=LLM_QUESTION,
        prompt=CONDENSE_QUESTION_PROMPT,
    )

    return chain


def qa_chain():
    system_template = """Use the following pieces of context to generate questions for user. 
    You need to ask question one at a time and formulate your question based on the answer of previous question by user. 
    You can stop asking the question when you have enough information in the history of messages about the context to create a high level WBS for the user.
    ----------------
    {context}
    Chat History:
    {chat_history}"""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        MessagesPlaceholder(variable_name="chat_history"),
        MessagesPlaceholder(variable_name="context"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

    template = (
        "Combine the chat history and follow up question into "
        "a standalone question. Chat History: {chat_history}"
        "Follow up question: {question}"
    )
    prompt = PromptTemplate.from_template(template)
    llm = OpenAI()
    question_generator_chain = LLMChain(llm=llm, prompt=prompt)

    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM_CHAT,
        chain_type="stuff",
        memory=CHAT_MEMORY,
        retriever=get_retriever(db_path=VECTOR_STORE_PATH),
        combine_docs_chain_kwargs={"prompt": CHAT_PROMPT},
        # question_chain=question_generator_chain,
    )

    # chain.memory = CHAT_MEMORY
    chain.question_generator = question_chain()
    # chain.combine_docs_chain = document_chain()
    # chain.get_chat_history = lambda h: h

    return chain


def qa_generation():
    templ = """You are a smart assistant designed to help project managers with project documents and requirements.
    Given a piece of text, you must come up with a series of questions.:
    ----------------
    {text}"""
    PROMPT = PromptTemplate.from_template(templ)

    chain = QAGenerationChain.from_llm(
        llm=LLM_QUESTION,
    )

    chain.input_key = "text"
    chain.output_key = "qa_pairs"
    chain.llm_chain = LLMChain(llm=LLM_QUESTION, prompt=PROMPT)

    # chain.question_generator = question_chain()

    return chain


# texts = process_files("./docs/Doc-1.pdf")
# create_vector_db(text=texts, vector_store_path=VECTOR_STORE_PATH, embeddings=EMBEDDINGS)
# ingest_files(file_path="./docs/Doc-1.pdf")

print(CHAT_MEMORY.chat_memory.messages)
while True:
    query = input("Enter your question: ")
    if query == "exit":
        break
    result = qa_chain().invoke(query)
    print(result["answer"])

# result = qa_chain().invoke("Tell me the WBS for the project")
# print(result)
