from dotenv import load_dotenv
load_dotenv()

import os
from fastapi import FastAPI, HTTPException, Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.pydantic_v1 import BaseModel, Field


GROQ_API_KEY = os.environ["GROQ_API_KEY"]

REDIS_URL = os.environ["REDIS_URL"]


app = FastAPI()


class Response(BaseModel):
    answer: str = Field(..., description="answer")


def Chat_W_LLM(session_id, user_input):

    def get_message_history(session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(session_id, url=REDIS_URL)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """"You are a tour guide agent. Your company name is Tour Buddy AI, Use history to generate different responses""",
            ),
            # MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    llm = ChatGroq(temperature=0, groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b-32768")

    runnable = prompt | llm.with_structured_output(schema=Response)


    with_message_history = RunnableWithMessageHistory(
        runnable,
        get_message_history,
        input_messages_key="input",
        history_messages_key="history",
        )

    response = with_message_history.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    # response = runnable.invoke({"ability" : ability, "input" : user_input})
    # print(response)
    return response.answer


@app.post("/QueryBot/{session_id}/{input}")
def returnChatResponse(
    session_id: str = Path(...),
    input: str = Path(...)):
    try:
        response = Chat_W_LLM(session_id=session_id, user_input=input)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main":
    import uvicorn
    uvicorn.run(app=app, host='127.0.0.0', port=8001)


