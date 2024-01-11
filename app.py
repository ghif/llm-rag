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
    chunks = dp.split_docs_into_chunks(docs)
    retriever = dp.create_vectorstore(chunks)

    chainer = rc.RAGChainer(retriever, 
        const.CONTEXTUALIZE_Q_SYSTEM_PROMPT, 
        const.QA_SYSTEM_PROMPT,
        llm_type="openai"
    )

    chat_history = []
    cl.user_session.set("chat_history", chat_history)
    cl.user_session.set("runnable", chainer.rag_chain)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")  # type: Runnable
    chat_history = cl.user_session.get("chat_history")

    response = cl.Message(content="")

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

    