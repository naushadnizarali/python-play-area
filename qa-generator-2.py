import openai
import os
import re
import pandas as pd

from sentence_extractor import process_files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

openai.api_key = os.environ["OPENAI_API_KEY"]

prompt_name = "question-imp.txt"


def load_prompt(filename, payload):
    with open("./prompts/%s" % filename, "r", encoding="utf-8") as infile:
        body = infile.read()
        body = body.replace("<<TEXT>>", payload)
        return body


def completion(
    prompt,
    engine="ft:gpt-3.5-turbo-1106:mobilelive-inc:irap-model:8mTN1w61",
    temp=0.3,
    top_p=0.95,
    tokens=200,
    freq_pen=0.5,
    pres_pen=0.5,
    stop=["\n\n"],
):
    try:
        response = openai.completions.create(
            model=engine,
            prompt=prompt,
            temperature=temp,
            max_tokens=tokens,
            top_p=top_p,
            frequency_penalty=freq_pen,
            presence_penalty=pres_pen,
            stop=stop,
        )
        text = response.choices[0].text.strip().splitlines()
        questions = ""
        for t in text:
            questions += re.sub("^\-", "", t).strip() + "\n"
        questions = questions.strip()
        return questions
    except Exception as oops:
        print("ERROR in completion function:", oops)


def process_texts(file_path):
    texts = process_files(file_path=file_path)
    question_gen = " ".join(texts)

    splitter_ques_gen = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    return document_ques_gen


if __name__ == "__main__":
    file_path = "./docs/Doc-4.pdf"
    doc_texts = process_texts(file_path)
    # doc_texts = doc_texts[:2]
    all_questions = ""

    for f in doc_texts:
        prompt = load_prompt(prompt_name, f.page_content)
        questions = completion(prompt)
        all_questions += questions.replace("\n\n", "").replace("\n", "")

    all_questions = all_questions.split("?")

    df = pd.DataFrame(all_questions, columns=["question"])
    base_folder = "./output/"
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)
    df.to_json("./output/questions.json", orient="records")
