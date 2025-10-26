from flask import Flask, render_template, jsonify, request
from flask_cors import CORS  # <— important to allow local HTML
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)
CORS(app)  

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_embeddings()

index_name = "medical-gpt"


docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})


chatModel = ChatGroq(model_name="llama-3.1-8b-instant")


prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt + "\n\nContext:\n{context}"),
    ("user", "{input}")
])

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("input", "")
    print("User:", msg)

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I couldn't find an answer.")
        print("Response:", answer)
        return jsonify({"answer": answer})
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"answer": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
