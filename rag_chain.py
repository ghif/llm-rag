from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from time import process_time

import docproc as dp
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

import constants as const

import csv

class RAGChainer:
    def __init__(self, 
        vectorstore, 
        llm_type: str="openai",
    ):
        """
        Args:
            vectorstore (Chroma): vectorstore
            llm_type (str): language model type

        """
        
        self.vectostore = vectorstore
        self.retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k":const.TOP_K}
        )

        if llm_type == "openai":
            self.llm = ChatOpenAI(
                model_name=const.MODEL_NAME, 
                temperature=const.TEMPERATURE,
                streaming=const.STREAMING
            )

        elif llm_type == "vertexai":
            self.llm = ChatVertexAI(
                model=const.GEMINI_MODEL_NAME
            )

        self.q_chain = None
        self.rag_chain = None
        self.with_history = False
    
    def init_with_history(self, contextualize_q_system_prompt: str, qa_system_prompt: str):
        """
        Initialize RAG chain with history

        Args:
            contextualize_q_system_prompt (str): contextualize question system prompt
            qa_system_prompt (str): question answering system prompt
            
        """
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        self.q_chain = contextualize_q_prompt | self.llm | StrOutputParser()

        self.rag_chain = self.create_rag_chain_with_history(
            qa_system_prompt
        )

    def init(self, qa_system_prompt: str):
        """
        Initialize RAG chain without history

        Args:
            qa_system_prompt (str): question answering system prompt
        """
        self.rag_chain = self.create_rag_chain(qa_system_prompt)
        

    def contextualized_question(self, input: dict):
        if input.get("chat_history"):
            return self.q_chain
        else:
            return input["question"]
    
    def create_rag_chain(self, qa_system_prompt:str):
        qa_prompt = PromptTemplate.from_template(qa_system_prompt)
        rag_chain = (
            {"context": self.retriever | dp.format_docs, "question": RunnablePassthrough()}
            | qa_prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def create_rag_chain_with_history(self, qa_system_prompt:str):
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
        rag_chain = (
            RunnablePassthrough.assign(
                context=self.contextualized_question | self.retriever | dp.format_docs
            )
            | qa_prompt
            | self.llm
        )
        return rag_chain

if __name__ == "__main__":
    docs = dp.get_docs_from_pdf(pdf_path=const.PDF_PATH)
    print(f"Number of pages from {const.PDF_PATH}: {len(docs)}")

    chunks = dp.split_docs_into_chunks(
        docs, 
        chunk_size=const.CHUNK_SIZE, 
        chunk_overlap=const.CHUNK_OVERLAP
    )

    vectorstore = dp.create_vectorstore(
        chunks,
        # embedding_type="openai",
        embedding_type="vertexai",
        persist_directory="db-sister-vertexai",
        collection_name=f"vstore_openai_sister_cs{const.CHUNK_SIZE}_co{const.CHUNK_OVERLAP}"
    )

    rc = RAGChainer(
        vectorstore, 
        # llm_type="openai"
        llm_type="vertexai"
    )

    if const.WITH_HISTORY:    
        rc.init_with_history(
            const.CONTEXTUALIZE_Q_SYSTEM_PROMPT,
            const.QA_SYSTEM_PROMPT
        )
        chat_history = []
    else:
        rc.init(
            const.RAG_PROMPT_NO_HISTORY
        )

    # Test prompt-response
    # queries = [
    #     "apa yang dimaksud dengan BKD?",
    #     "Bagaimana cara Pengelola BKD PTN menambah Periode BKD?",
    #     "bagaimana mendaftar ke program kampus merdeka?"
    # ]
        
    # Positive queries
    queries = [
        "apa kepanjangan dari BKD?",
        "apa yang dimaksud dengan Beban Kerja Dosen?",
        "apakah yang dimaksud dengan Asesor BKD?",
        "Bagaimana cara untuk menjadi Asesor BKD?",
        "apakah yang dimaksud dengan NIRA BKD?",
        "Bagaimana jika saya memiliki NIRA ganda?",
        "bagaimana caranya mendaftar kampus merdeka?"
    ]
        
    
     # Write query-response to CSV
    csv_filename = "testing_sister_openai-embeddings.csv"
    fields = ["Query", "AI Response"]
    csv_fhandler = open(csv_filename, 'w')
    csvwriter = csv.writer(csv_fhandler, delimiter=';')

    # writing the fields
    csvwriter.writerow(fields)

    for query in queries:
        # Check retrieved docs
        retrieved_docs = rc.retriever.invoke(query)
        for i, rdoc in enumerate(retrieved_docs):
            print(f"[{i}] {rdoc.page_content}")
            print("\n")
        print(f"Retriever docs: {len(retrieved_docs)}")

        print(f"\nQuery: {query}\n")

        start_t = process_time()
        if const.WITH_HISTORY:
            ai_msg = rc.rag_chain.invoke(
                {
                    "question": query,
                    "chat_history": chat_history
                }
            )
            chat_history.extend([HumanMessage(content=query), ai_msg])

        else:
            ai_msg = rc.rag_chain.invoke(query)

        elapsed_t = process_time() - start_t

        response = ai_msg if isinstance(ai_msg, str) else ai_msg.content

        print(f"\nResponse ({elapsed_t:.2f}secs): \n")
        print(f"{response}\n")

        csvwriter.writerow([query, response])

    # end for
        
    csv_fhandler.close()