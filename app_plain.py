from langchain.schema.runnable.config import RunnableConfig
import chainlit as cl

import plain_chain as pc
import constants as const


@cl.on_chat_start
async def on_chat_start():
    ch = pc.Chainer(llm_type="vertexai")
    ch.init(const.STANDARD_SYSTEM_PROMPT)
    
    cl.user_session.set("chainer", ch.chain)


@cl.on_message
async def on_message(message: cl.Message):
    chainer = cl.user_session.get("chainer")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in chainer.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()