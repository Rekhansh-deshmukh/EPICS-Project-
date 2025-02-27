from flask import Flask, render_template, jsonify, request

from src.helper import downlaod_hugging_face_embeddings
from langchain_pineapple import pineconeVectorSpace
from langchain_ollama import Ollama
from langchain.chain import create_retrival_chain
from langchain.chain.combine_documents import create_stuff_documents
from langchain_core.prompt import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

embedding = downlaod_hugging_face_embeddings()

index_name = "synthmind"

docsearch = PineconeVectorStore.from_documents(documents=text_chunks, embedding=embedding, index_name=index_name)
retriver = docsearch.as_retriever(search_type="similarity",search_kwarg={"k":3})
llm = ChatOllama(model="deepseek-coder-v2")
prompt = ChatPromptTemplate.from_messages(
    [("system",system_prompt),
     ("human","{input}"),
    ]
)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('message')
    # Here you would call the Ollama model to get a response
    response = "This is a placeholder response."  # Replace with actual model call
    return jsonify({'response': response})


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080,debug=True)
