from langchain.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Extract text from the PDF files
def load_pdfs_file(data):
    loader = DirectoryLoader(
        data, 
        glob ="*.pdf",
        loader_cls=PyPDFLoader
    )
    
    doc = loader.load()
    return doc

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

def text_splitter(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        length_function=len
    )
    texts = text_splitter.split_documents(docs)
    return texts

def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings


