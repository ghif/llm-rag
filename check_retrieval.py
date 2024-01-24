from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002")

db = Chroma(
    persist_directory="db-sister",
    collection_name="vstore_openai_sister", 
    embedding_function=embedding_function
)

retriever = db.as_retriever(search_type="mmr")

query = "apa yang dimaksud dengan BKD?"
print(f"\nQuery: {query}\n")
response = retriever.get_relevant_documents(query)

for i, res in enumerate(response):
    print("\n")
    print(f"[{i}] Doc: {res.page_content}")
print(f"Number of response: {len(response)}")