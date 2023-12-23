
from flask import Flask, render_template, request, jsonify
from src.utils import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

#download embeddings
embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
pinecone.init(      
	api_key='c3701bf9-f5f8-4b3d-82da-118a22b743bf',      
	environment='gcp-starter')      
index_name = 'medibot'

#using existing index
docsearch = Pinecone.from_existing_index(index_name, embeddings)

#Create Prompt Template
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain_type_kwargs={"prompt": PROMPT}

#initialize 
chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})

qa=RetrievalQA.from_chain_type(
    LLM=llm,
    chain_type = "stuff",
    retriever = docsearch.as_retriever(search_kwargs={'K': 2,}),
    return_source_documents = True,
    chain_type_kwargs= chain_type_kwargs)

#Make a default route for flask
@app.route("/")
def index():
    return render_template('chat.html')

#Final Route
@app.route("/get", methods = ["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa({"query": input})
    print ("Response :", result["result"])
    return str(result["result"])
    

#initialize flask
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 8080, debug=True)
