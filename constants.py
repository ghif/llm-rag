RAG_PROMPT_NO_HISTORY = """
Use only the pieces of context below to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use five sentences maximum and keep the answer as concise as possible.
Always say the following sentences at the end of the answer:

\n
Terima kasih atas pertanyaannya. 
Apakah jawaban kami sudah membantu?

{context}

Question: {question}

Helpful Answer:
"""

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

QA_SYSTEM_PROMPT = """You are an assistant for question-answering tasks. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise. \
Always say the following sentences at the end of the answer:

\n
Terima kasih atas pertanyaannya. 
Apakah jawaban kami sudah membantu?

Use only the following pieces of retrieved context to answer the question. \

{context}"""


PDF_PATH = "data/SSD_SISTER_BKD.pdf"

MODEL_NAME = "gpt-3.5-turbo-1106"
# MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.
STREAMING = True

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

WITH_HISTORY = False

SENTENCE_TRANSFORMER_MODEL_NAME = "all-MiniLM-L6-v2"