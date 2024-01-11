from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from time import process_time

import docproc as dp
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage

import constants as const

class RAGChainer:
    def __init__(self, 
        retriever, 
        contextualize_q_system_prompt: str,
        qa_system_prompt: str,
        llm_type: str="openai",
    ):
        
        self.retriever = retriever

        if llm_type == "openai":
            self.llm = ChatOpenAI(
                model_name=const.MODEL_NAME, 
                temperature=const.TEMPERATURE,
                streaming=const.STREAMING
            )

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

    def contextualized_question(self, input: dict):
        if input.get("chat_history"):
            return self.q_chain
        else:
            return input["question"]
    
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
    chunks = dp.split_docs_into_chunks(docs)
    retriever = dp.create_vectorstore(chunks)

    rc = RAGChainer(retriever, 
        const.CONTEXTUALIZE_Q_SYSTEM_PROMPT, 
        const.QA_SYSTEM_PROMPT,
        llm_type="openai"
    )

    # Test prompt-response
    queries = [
        # "bagaimana mendaftar ke program kampus merdeka?",
        "jelaskan tentang BKD",
        "bagaimana cara mengisinya di SISTER?",
        "bagaimana mendaftar ke program kampus merdeka?"
    ]

    chat_history = []
    for query in queries:
        print(f"\nQuery: {query}\n")

        # # Check retrieved docs
        # retrieved_docs = retriever.invoke(query)
        # for i, rdoc in enumerate(retrieved_docs):
        #     print(f"[{i}] {rdoc.page_content[:100]}")
        # print(f"Retriever docs: {len(retrieved_docs)}")

        start_t = process_time()
        ai_msg = rc.rag_chain.invoke(
            {
                "question": query,
                "chat_history": chat_history
            }
        )
        elapsed_t = process_time() - start_t
        print(f"\nResponse ({elapsed_t:.2f}secs): \n")
        print(f"{ai_msg.content}\n")

        chat_history.extend([HumanMessage(content=query), ai_msg])