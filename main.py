from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse

from dotenv import load_dotenv

import os

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import chromadb


app = FastAPI()
load_dotenv()

templates = Jinja2Templates(directory="templates")


CHUNK_SIZE = 300
CHUNK_OVERLAP = 75
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


class Wiki(BaseModel):
    wiki_url : str


def load_wiki_article(url: str):
    try:
        loader = WebBaseLoader(
            web_path=url
        )
        docs = loader.load()
        print("Wiki Loaded Successfuly")
    except Exception as e:
        raise Exception(e)
    
    return docs

def wiki_text_splitter(docs):

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE, 
    chunk_overlap=CHUNK_OVERLAP)

    # Make splits
    splits = text_splitter.split_documents(docs)

    return splits

async def create_db(splits):

    chroma_instance = Chroma(persist_directory='db/')
    delete_db_documents(chroma_instance)

    vectorstore = Chroma.from_documents(documents=splits, 
                                    embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
                                    persist_directory='db/')
    #collection_metadata={"hnsw:M": 2024,"hnsw:ef": 500}

async def get_retriever():
    # load chroma db
    chroma_instance = Chroma(persist_directory='db/', embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    retriever = chroma_instance.as_retriever()
    
    return retriever


def delete_db_documents(chroma_instance):

    chroma_db =  chromadb.PersistentClient('db/')
    chroma_db.reset()

    # ids = chroma_instance.get()['ids']
    # if len(ids) > 0:

    #     print("count before", chroma_instance._collection.count())
    #     chroma_instance._collection.delete(ids=ids)
    # print("count after", chroma_instance._collection.count())


@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/send", response_class=HTMLResponse)
async def read_wiki_url(request: Request, wiki: Wiki):

    url = wiki.wiki_url
    error_message = ""
    docs = load_wiki_article(url)
    splits = wiki_text_splitter(docs)
    await create_db(splits)
    #print('Finished execution')
    
    
    return templates.TemplateResponse("chat.html", {"request": request})

    # try:

    #     # docs = load_wiki_article(url)
    #     # splits = wiki_text_splitter(docs)
    #     # await create_db(splits)
    #     print('Finished execution')
    #     #return templates.TemplateResponse("chat.html", {"request": request})
    #     return RedirectResponse(url="/chat")
        
    # except Exception as e:
    #     rerror_message = str(e)
    #     print(error_message)
    #     return templates.TemplateResponse("index.html", {"request": request, "error_message": error_message})
    

@app.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    # Add any necessary context variables to the context dictionary
    print("Chat endpoint reached")
    context = {"request": request}
    return templates.TemplateResponse("chat.html", context)



@app.post("/chat", response_class=HTMLResponse)
async def ask_respond(request: Request):

    request_body = await request.json()
    
    # Access the value of the "question" field from the request body
    user_question = request_body.get("question")

    retriever = await get_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7) 

    template = """Answer the question based only on the following context:
    {context} and when you dont know the answer say I dont have  information about AND MENTION WHAT THE USER AKED YOU ABOUT
    but you can be kind for example when people are greeting you

    Question: {question}
    """
    # template = """Answer the question based only on the following context:
    # {context}
    # Refer to the {history} of the conversation -if available- to better understand the question

    # Question: {question}
    # """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("human_question:", user_question)
    answer = rag_chain.invoke(user_question)

    print("ai_answer:", answer)

    return JSONResponse(content={"answer": answer})





# def x():
#     from langchain.chains import create_retrieval_chain
#     from langchain.chains.combine_documents import create_stuff_documents_chain

#     qa_system_prompt = """You are an assistant for question-answering tasks. \
#     Use the following pieces of retrieved context to answer the question. \
#     If you don't know the answer, just say that you don't know. \
#     Use three sentences maximum and keep the answer concise.\

#     {context}"""
#     qa_prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", qa_system_prompt),
#             MessagesPlaceholder("chat_history"),
#             ("human", "{input}"),
#         ]
#     )
#     question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

#     rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)




#%%
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_openai import ChatOpenAI
# from langchain.prompts import ChatPromptTemplate
# import os
# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# # Initialize Chroma with the saved database
# chroma_instance = Chroma(persist_directory='.', embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))


# # %%
# ids = chroma_instance.get()['ids']
# print("count before", chroma_instance._collection.count())


# #chroma_instance._collection.delete(ids=ids)
# print("count after", chroma_instance._collection.count())
# #  %%
# import chromadb

# chroma_db =  chromadb.PersistentClient('.')
# chroma_db.reset()

# chroma_db.delete_collection('langchain')



#%%

# # Prompt
# template = """Answer the question based only on the following context:
# {context} and when you dont know the answer say 'Sorry I can't answer that'

# Question: {question}
# """

# prompt = ChatPromptTemplate.from_template(template)
# prompt

# retriever = chroma_instance.as_retriever()
# llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0) 

# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough

# rag_chain = (
#     {"context": retriever, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# rag_chain.invoke("who founded vodafone?")
# %%
