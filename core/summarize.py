from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableAssign,RunnableLambda
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.environ["MISTRAL_API_KEY"],
        temperature=0.3
    )
def split_transcript(transcript: str)->list:
    splitter=RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)
def summarize(transcript: str)->str:
    llm=get_llm()
    split=split_transcript(transcript)
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcription concisely"),
        ("human", "{text}")
    ])
    map_chain = map_prompt | llm | StrOutputParser()
    chunks=split_transcript(transcript)
    chunk_summaries=[map_chain.invoke({"text":chunk})for chunk in chunks]
    final_prompt=ChatPromptTemplate.from_messages([
        ("system", """you are an expert meeting summarizer combine these partial summaries into a coherent final summary of the entire meeting. don't add any additional information ,just provide summary of the meetin """),
        ("human", "{text}")
    ])
    final_prompt=(RunnablePassthrough()|RunnableLambda(lambda x: "\n\n".join(x))|final_prompt|llm|StrOutputParser())
    
    return final_prompt.invoke({"text":chunk_summaries})
def generate_title(transcript:str)->str:
    llm=get_llm()
    title_prompt=ChatPromptTemplate.from_messages([
        ("system", "Based on the meeting transcript generate a title for the meeting.max(8 words)"),
        ("human", "{text}")
    ])
    title_chain=(RunnablePassthrough()|title_prompt|llm|StrOutputParser())
    return title_chain.invoke({"text":transcript[:2000]})
    
    
