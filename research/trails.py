import os
os.chdir("../")
# print(os.getcwd())

from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Extract text from the PDF files
def load_pdfs_file(data):
    loader = DirectoryLoader(
        data, 
        glob ="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    doc = loader.load()
    return doc

extracted_docs = load_pdfs_file("data")
# print(extracted_docs)
# print(len(extracted_docs))


from typing import List
from langchain.schema import Document

def filter_to_minimal_docs(docs: List[Document])->List[Document]:
    minimal_docs = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content, 
                metadata={"source": src}
            )
        )
    return minimal_docs
    

minimal = filter_to_minimal_docs(extracted_docs)
# print(minimal)

def text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    texts = text_splitter.split_documents(docs)
    return texts

texts = text_splitter(minimal)
# print(texts)

from langchain_huggingface import HuggingFaceEmbeddings

def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings

embeddings = download_embeddings()
# print(embeddings.embed_query("Hello World"))


from dotenv import load_dotenv
import os
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY 
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)

from pinecone import ServerlessSpec
index_name = "medicalgpt"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore

# docsearch = PineconeVectorStore(
#     embedding=embeddings,
#     index_name=index_name,
# )
# docsearch.add_documents(documents = texts)


# Loading existing index
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name,
)

retriver = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})
# print(retriver.get_relevant_documents("What is diabetes?"))

from langchain_groq import ChatGroq

chatModel = ChatGroq(
    model_name="deepseek-r1-distill-llama-70b"
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

system_prompt = """
You are a helpful medical assistant. 
Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Do NOT reveal your internal chain-of-thought, deliberations, or any <think> tags.
Only return the final concise answer (maximum 3 sentences). If you don't know, say you don't know.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt + "\n\nContext:\n{context}"),
    ("user", "{input}")
])

question_answering_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriver, question_answering_chain)

response = rag_chain.invoke({"input": "What is diabetes?"})
import re

response = rag_chain.invoke({"input": "What is diabetes?"})
raw_answer = response.get("answer") or str(response)
clean_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.S).strip()
print(clean_answer)