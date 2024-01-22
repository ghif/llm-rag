from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

import rag_chain as rc
import constants as const
import docproc as dp

import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
    docs = dp.get_docs_from_pdf(pdf_path=const.PDF_PATH)
    chunks = dp.split_docs_into_chunks(
        docs, 
        chunk_size=const.CHUNK_SIZE,
        chunk_overlap=const.CHUNK_OVERLAP
    )
    # vectorstore = dp.create_vectorstore(
    #     chunks,
    #     embedding_type="sentence_transformer",
    # )
    vectorstore = dp.create_vectorstore(
        chunks,
        embedding_type="openai",
        persist_directory="db-sister",
        collection_name="vstore_openai_sister"
    )

    chainer = rc.RAGChainer(
        vectorstore, 
        llm_type="openai"
    )

    if const.WITH_HISTORY:
        chainer.init_with_history(
            const.CONTEXTUALIZE_Q_SYSTEM_PROMPT,
            const.QA_SYSTEM_PROMPT
        )
        chat_history = []
        cl.user_session.set("chat_history", chat_history)
    else:
        chainer.init(
            const.RAG_PROMPT_NO_HISTORY
        )

    cl.user_session.set("chainer", chainer)


@cl.on_message
async def on_message(message: cl.Message):
    chainer = cl.user_session.get("chainer")  
    runnable = chainer.rag_chain

    print(f"\n\nQuery: {message.content}\n\n")
    retrieved_docs = chainer.retriever.invoke(message.content)

    for i, rdoc in enumerate(retrieved_docs):
        print(f"\n\n[{i}] {rdoc.page_content[:1000]}")
        print(f"Retriever docs: {len(retrieved_docs)}\n\n")
                                    

    response = cl.Message(content="")

    if const.WITH_HISTORY:
        chat_history = cl.user_session.get("chat_history")
        async for chunk in runnable.astream(
            {
                "question": message.content,
                "chat_history": chat_history,
            },
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await response.stream_token(chunk.content)

        await response.send()

        chat_history.extend(
            [
                HumanMessage(content=message.content), 
                AIMessage(content=response.content)
            ]
        )
    else:
        async for chunk in runnable.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await response.stream_token(chunk)

        await response.send()

    

    