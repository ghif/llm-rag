from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

import constants as const

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


class Chainer:
    def __init__(self, 
        llm_type: str="openai",

    ):
        if llm_type == "openai":
            self.llm = ChatOpenAI(
                model_name=const.MODEL_NAME, 
                temperature=const.TEMPERATURE,
                streaming=const.STREAMING
            )
        
        self.chain = None
        
    def init(self, system_prompt:str):
        self.chain = self.create_chain(system_prompt)
    

    def create_chain(self, system_prompt:str):
        prompt = PromptTemplate.from_template(system_prompt)
        chain = (
            {"question": RunnablePassthrough()} 
            | prompt | self.llm | StrOutputParser()
        )
        return chain

if __name__ == "__main__":
    ch = Chainer(llm_type="openai")
    ch.init(const.STANDARD_SYSTEM_PROMPT)

    query = "apa yang dimaksud dengan NIRA BKD?"
    print(f"\nQuery: {query}")
    ai_msg = ch.chain.invoke({
        "question": query
    })
    print(f"Response: {ai_msg}")

