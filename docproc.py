from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

def get_docs_from_pdf(pdf_path:str=None):
    """
    Get docs from pdf
    Args:
        pdf_path (str): path to pdf file
    Returns:
        docs (list): list of pages extracted from the pdf
    """
    if pdf_path is None:
        pdf_path = "data/cerita-rakyat-nusantara2.pdf"

    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    return docs

def split_docs_into_chunks(
        docs:list,
        chunk_size:int=1000, 
        chunk_overlap:int=200
    ):
    """
    Split docs into chunks
    Args:
        chunk_size (int): size of each chunk
        chunk_overlap (int): overlap between chunks
    Returns:
        chunks (list): list of chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def create_vectorstore(chunks):
    """
    Create vectorstore from chunks
    Args:
        chunks (list): list of chunks
    Returns:
        retriever (Chroma): retriever
    """
    vectorstore = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    pdf_path = "data/cerita-rakyat-nusantara2.pdf"
    docs = get_docs_from_pdf(pdf_path=pdf_path)
    print(f"Number of pages: {len(docs)}")

    chunks = split_docs_into_chunks(docs)
    print(f"Number of chunks: {len(chunks)}")
    