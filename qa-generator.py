from fastapi import (
    FastAPI,
    Form,
    Request,
    Response,
    File,
    Depends,
    HTTPException,
    status,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from langchain_community.llms import CTransformers
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
import os
import json
import time
import uvicorn
import aiofiles
from PyPDF2 import PdfReader
import csv

from sentence_extractor import process_files

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


def load_llm():
    # Load the locally downloaded model here
    # llm = CTransformers(
    #     model="/Users/naushadali/.cache/lm-studio/models/TheBloke/Llama-2-7B-GGUF/llama-2-7b.Q4_K_M.gguf",
    #     model_type="mistral",
    #     max_new_tokens=1048,
    #     temperature=0.3,
    # )

    llm = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        model="ft:gpt-3.5-turbo-1106:mobilelive-inc:irap-model:8mTN1w61",
        temperature=0.3,
        max_tokens=4096,
    )
    return llm


def file_processing(file_path):

    # Load data from PDF
    # loader = PyPDFLoader(file_path)
    # data = loader.load()

    question_gen = ""

    # for page in data:
    #     question_gen += page.page_content

    texts = process_files(file_path=file_path)
    question_gen = " ".join(texts)
    # create_vector_db(text=texts, vector_store_path=VECTOR_STORE_PATH, embeddings=EMBEDDINGS)

    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    # splitter_ans_gen = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

    # document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    # return document_ques_gen, document_answer_gen
    return document_ques_gen


def llm_pipeline(file_path):

    # document_ques_gen, document_answer_gen = file_processing(file_path)
    document_ques_gen = file_processing(file_path)

    llm_ques_gen_pipeline = load_llm()

    prompt_template = """
    You are an expert at creating questions based on materials and documentation.
    Your goal is to prepare a set of questions.
    You do this by asking questions about the text below:

    ------------
    {text}
    ------------

    Create questions that will prepare the end users to create a comprehensive WBS out of the context. You need to focus on techincal and business perspective.
    Make sure not to lose any important information.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(
        template=prompt_template, input_variables=["text"]
    )

    refine_template = """
    You are an expert at creating questions based on material and documentation.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(
        llm=llm_ques_gen_pipeline,
        chain_type="refine",
        verbose=True,
        question_prompt=PROMPT_QUESTIONS,
        refine_prompt=REFINE_PROMPT_QUESTIONS,
    )

    ques = ques_gen_chain.run(document_ques_gen)

    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-mpnet-base-v2"
    # )

    # vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    # llm_answer_gen = load_llm()

    ques_list = ques.split("\n")
    filtered_ques_list = [
        element
        for element in ques_list
        if element.endswith("?") or element.endswith(".")
    ]

    # answer_generation_chain = RetrievalQA.from_chain_type(
    #     llm=llm_answer_gen, chain_type="stuff", retriever=vector_store.as_retriever()
    # )

    # return answer_generation_chain, filtered_ques_list
    return filtered_ques_list


def get_csv(file_path):
    # answer_generation_chain, ques_list = llm_pipeline(file_path)
    ques_list = llm_pipeline(file_path)
    base_folder = "static/output/"
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    output_file = base_folder + "QA.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])  # Writing the header row

        for question in ques_list:
            print("Question: ", question)
            # answer = answer_generation_chain.run(question)
            # print("Answer: ", answer)
            print("--------------------------------------------------\n\n")

            # Save answer to CSV file
            csv_writer.writerow([question, ""])
    return output_file


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload")
async def chat(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    base_folder = "static/docs/"
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    pdf_filename = os.path.join(base_folder, filename)

    async with aiofiles.open(pdf_filename, "wb") as f:
        await f.write(pdf_file)
    response_data = jsonable_encoder(
        json.dumps({"msg": "success", "pdf_filename": pdf_filename})
    )
    res = Response(response_data)
    return res


@app.post("/analyze")
async def chat(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)
    response_data = jsonable_encoder(json.dumps({"output_file": output_file}))
    res = Response(response_data)
    return res


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
