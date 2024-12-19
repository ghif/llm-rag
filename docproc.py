from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import constants as const

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
    # return pdf_texts

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
        # separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    text_chunks = text_splitter.split_documents(docs)
    # token_splitter = SentenceTransformersTokenTextSplitter(
    #     chunk_overlap=0,
    #     model_name=const.SENTENCE_TRANSFORMER_MODEL_NAME
    # )
    # token_chunks = token_splitter.split_documents(text_chunks)

    # chunks = token_chunks
    chunks = text_chunks
    return chunks

def create_vectorstore(
        chunks,
        embedding_type="openai",
        persist_directory="db",
        collection_name="vstore_sister_ssd"
    ):
    """
    Create vectorstore from chunks
    Args:
        chunks (list): list of chunks
    Returns:
        retriever (Chroma): retriever
    """

    if embedding_type == "openai":
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
    elif embedding_type == "sentence_transformer":
        embedding_function = SentenceTransformerEmbeddings(
            model_name=const.SENTENCE_TRANSFORMER_MODEL_NAME
        )
    elif embedding_type == "vertexai":
        embedding_function = VertexAIEmbeddings(model="text-embedding-004")

    print(f"Creating vectorstore with embedding type: {embedding_type}")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    return vectorstore

# def load_vectorstore(
#         persist_directory="db",
#         collection_name="vstore_sister_ssd"
#     )

def load_vectorstore(
        persist_directory:str,
        collection_name:str,
        embedding_type:str="openai"
    ):
    if embedding_type == "openai":
        embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")
    elif embedding_type == "vertexai":
        embedding_function = VertexAIEmbeddings(model="text-embedding-004")

    db = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name, 
        embedding_function=embedding_function
    )
    return db

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    # pdf_path = "data/cerita-rakyat-nusantara2.pdf"
    pdf_path = "data/SSD_SISTER_BKD.pdf"
    docs = get_docs_from_pdf(pdf_path=pdf_path)
    print(f"Number of pages: {len(docs)}")

    chunks = split_docs_into_chunks(
        docs,
        chunk_size=const.CHUNK_SIZE,
        chunk_overlap=const.CHUNK_OVERLAP
    )
    print(f"Number of chunks: {len(chunks)}")

    print(f"docs 0: {docs[0].page_content[:100]}")
    print(f"chunks 0: {chunks[0].page_content[:100]}")

    # vectorstore = create_vectorstore(
    #     chunks,
    #     embedding_type="sentence_transformer",
    # )

    vectorstore = create_vectorstore(
        chunks,
        embedding_type="vertexai",
        persist_directory="db-sister-vertexai",
        collection_name="vstore_strans_sister_vertexai"
    )

    # query it
    # query = "Apa yang dimaksud dengan Beban Kerja Dosen?"
    query = "apa yang dimaksud dengan BKD?"
    # query = "Bagaimana cara Pengelola BKD PTN menambah Periode BKD?"
    # query = "bagaimana mendaftar ke program kampus merdeka?"
    # queries = [
    #     "apa yang dimaksud dengan BKD?",
    #     "Bagaimana cara Pengelola BKD PTN menambah Periode BKD?",
    #     "bagaimana mendaftar ke program kampus merdeka?"
    # ]
    print(f"\nQuery: {query}\n")
    response = vectorstore.similarity_search_with_score(query)

    for i, res in enumerate(response):
        print("\n")
        print(f"[{i}] (Score: {res[1]}) Doc: {res[0].page_content}")
    print(f"Number of response: {len(response)}")


    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    response2 = retriever.invoke(query)
    # print(response2)

    for i, res in enumerate(response2):
        # print(f"Score: {res.score}")
        print("\n")
        print(f"[{i}] Doc: {res.page_content}")

    print(f"Number of response 2: {len(response2)}")

    collection = vectorstore.get()
    print(f"Number of collection: {len(collection)}")