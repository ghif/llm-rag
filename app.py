from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage, AIMessage

import rag_chain as rc
import constants as const
import docproc as dp

import chainlit as cl

import constants as const

@cl.on_chat_start
async def on_chat_start():
    vectorstore = dp.load_vectorstore(
        "db-sister",
        f"vstore_openai_sister_cs{const.CHUNK_SIZE}_co{const.CHUNK_OVERLAP}",
        embedding_type="openai",
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

    

    