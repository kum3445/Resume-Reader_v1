from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from secretkey import openapi_key
import gradio as gr
from typing_extensions import Concatenate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()
openai_api_key=os.getenv("api_key")

filename=r'C:\Users\mjyothivenkatasai\Desktop\GENAI\pdf reader\resume.pdf'


def pdf(filename):
    pdfreader= PdfReader(filename)
    # read text from pdf
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # We need to split the text using Character Text Split such that it sshould not increse token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 3800,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    return texts


def main(query):
    texts=pdf(filename)
    embeddings = OpenAIEmbeddings(openai_api_key=openapi_key)

    document_search = FAISS.from_texts(texts, embeddings)
    chain = load_qa_chain(OpenAI(openai_api_key=openapi_key), chain_type="stuff")
    docs = document_search.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response


iface = gr.Interface(

    fn=main,

    inputs=[

        gr.Textbox(lines=5, label="Prompt"),

        #gr.inputs.File(label="Data File (Optional)"),

    ],

    outputs=gr.Textbox(label="Response")

)
 
# Run the app

iface.launch(share=True)