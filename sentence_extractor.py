import spacy
import os
import pdfplumber
import re

from nltk.tokenize import sent_tokenize
from keywords import keyword_list
from spacy.matcher import Matcher
from langchain_community.vectorstores import FAISS

nlp = spacy.load("en_core_web_sm")


# Function to clean up the text
def clean_text(text):
    text = re.sub(r"\n\s*\n", " ", text)  # Remove multiple newlines
    text = re.sub(r"[^a-zA-Z0-9,.!?;\s]", "", text)  # Remove special characters
    text = re.sub(r"\b\d+\b", "", text)  # Remove orphan digits
    text = re.sub(r"\s{2,}", " ", text)  # Replace multiple spaces with single space
    text = re.sub(
        r"Page \d+/\d+", "", text
    )  # Remove page numbers (assuming a "Page X of Y" format)
    text = re.sub(
        r"http[s]?://\S+", "", text
    )  # Remove URLs (assuming http or https pattern)
    text = re.sub(
        r"[^a-zA-Z0-9,.!?;\n ]+", "", text
    )  # Remove special characters (excluding alphanumeric and basic punctuation)
    text = re.sub(
        r"\b\d+\b", "", text
    )  # Remove orphan digits (digits that are on their own, not part of a word or larger number)
    text = re.sub(
        r"\n\s*\n", "\n\n", text
    )  # Strip extra whitespace between paragraphs/sentences
    text = re.sub(
        r"\s+", " ", text
    ).strip()  # Remove newline characters, multiple spaces and trim leading/trailing spaces
    lines = [
        line.strip() for line in text.splitlines()
    ]  # Strip leading/trailing whitespace on each line
    lines = [
        line for line in lines if line
    ]  # Remove lines that are empty after stripping whitespace
    text = "\n".join(lines)  # Join the lines back into a single string
    text = re.sub(
        r" +", " ", text
    )  # Optionally, if you want to ensure single spacing between words everywhere
    text = text.strip()  # Remove leading/trailing spaces
    return text


# Function to check for table of contents
def is_table_of_contents_or_appendix(text):
    # Basic keyword check (can be made more complex if necessary)
    if re.search(r"\bcontents\b", text, re.IGNORECASE):
        return True
    if re.search(r"\bappendix\b", text, re.IGNORECASE):
        return True
    # Check for patterns like "some text ... 1"
    if re.search(r"(?<=\S)\.\s*\d+\s*$", text, re.MULTILINE):
        return True
    return False


def extract_clean_text_from_pdf(pdf_path):
    # Open the PDF file using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        meaningful_text = ""
        for page in pdf.pages:
            # You can analyze the page layout here to determine regions to extract text from
            # Example: page.crop((x, y, width, height)).extract_text()
            text = page.extract_text()
            if text and not is_table_of_contents_or_appendix(text):
                text = clean_text(text)
                meaningful_text += text + " "

    return meaningful_text


# Define a function to convert keywords into spaCy patterns
def create_matcher_patterns(keywords):
    matcher = Matcher(nlp.vocab)
    patterns = [[{"LOWER": keyword.lower()}] for keyword in keywords]
    # Add the patterns to the matcher
    matcher.add("KeywordMatcher", patterns)
    return matcher


# Define a function to extract sentences that contain keywords using spaCy
def extract_relevant_sentences_with_spacy(text, matcher):
    # Process the document text with spaCy
    doc = nlp(text)

    # Find matches in the doc
    matches = matcher(doc)

    # Dictionary to hold sentences keyed by keyword
    relevant_info = {}
    raw_sentences = []

    # Loop through the matches
    for match_id, start, end in matches:
        # Get the matching span
        span = doc[start:end]
        sentence = span.sent  # The sentence containing the match
        match_text = span.text.lower()

        # Capture the sentence for each keyword
        if match_text not in relevant_info:
            relevant_info[match_text] = []
        relevant_info[match_text].append(sentence.text.strip())
        raw_sentences.append(sentence.text.strip())

    return relevant_info, raw_sentences


def process_files(file_path: os.PathLike):
    print(f"Processing {file_path} ...")
    pdf_text = extract_clean_text_from_pdf(file_path)
    matcher = create_matcher_patterns(keyword_list)
    relevant_sentences = extract_relevant_sentences_with_spacy(pdf_text, matcher)

    print(f"Found {len(relevant_sentences[1])} relevant sentences.")

    return relevant_sentences[1]


def create_vector_db(text, embeddings, vector_store_path):
    vectordb = FAISS.from_texts(text, embedding=embeddings)
    vectordb.save_local(vector_store_path)


# if __name__ == "__main__":
# pdf_path = "./docs/Doc-3.pdf"
# relevant_sentences = process_files(pdf_path)

# print(relevant_sentences[1])

# # For example, output the relevant_sentences related to a specific keyword
# for keyword, sentences in relevant_sentences[0].items():
#     print(f"Keyword: {keyword}")
#     for sentence in sentences:
#         print(f"- {sentence}")
#     print("\n")
