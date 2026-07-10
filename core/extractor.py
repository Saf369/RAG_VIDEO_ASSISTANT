#extract Actionable items,decision,questions from transcript
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
import os
from dotenv import load_dotenv

load_dotenv()
def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        api_key=os.environ["MISTRAL_API_KEY"],
        temperature=0.2
    )
def build_chain(system_prompt: str):
    """Factory that returns a runnable chain for a given system prompt."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt.strip()),
        ("human", "{text}")
    ])
    return (RunnablePassthrough() | prompt | llm | StrOutputParser())

def sample_transcript(transcript: str, chunk_size: int = 800) -> str:
    """Sample beginning, middle, and end to cover the full video."""
    n = len(transcript)
    if n <= chunk_size * 3:
        return transcript
    start = transcript[:chunk_size]
    mid = transcript[n // 2 - chunk_size // 2 : n // 2 + chunk_size // 2]
    end = transcript[-chunk_size:]
    return f"{start}\n\n[...middle of video...]\n\n{mid}\n\n[...end of video...]\n\n{end}"

def build_action_item_chain(transcript: str) -> str:
    chain = build_chain("""
        You are an expert video content analyst.
        Your job is to extract key takeaways from this video transcript — things the viewer
        should learn, consider, or act on after watching.
        This includes:
          - Key lessons or insights the viewer should remember
          - Warnings, risks, or things the viewer should be aware of
          - Strategies, tactics, or techniques suggested (explicitly or implicitly)
          - Tools, resources, products, or concepts mentioned worth exploring
          - Calls to action from the creator (e.g. subscribe, comment, share)
          - Perspectives or mindset shifts the creator encourages
        Be generous — if the creator implies something is worth paying attention to, include it.
        Format your reply as a numbered list (e.g. 1. item, 2. item), one item per line.
        If there are truly no takeaways, respond with: "no actionable takeaways found"
    """)
    return chain.invoke({"text": sample_transcript(transcript)})

def extract_actionable_items(transcript: str) -> list:
    return build_action_item_chain(transcript).split("\n")


def build_questions_chain(transcript: str) -> str:
    chain = build_chain("""
        You are an expert video content analyst.
        Your job is to extract meaningful questions from this video transcript.
        This includes:
          - Questions the creator explicitly raises but does not fully answer
          - Open-ended questions posed to the viewer to think about
          - Rhetorical questions used to highlight a problem or tension
          - Unresolved debates or dilemmas mentioned in the video
        Do NOT include questions that are clearly answered within the transcript.
        Format your reply as a numbered list (e.g. 1. question, 2. question), one per line.
        If there are no such questions, respond with: "no open questions found"
    """)
    return chain.invoke({"text": sample_transcript(transcript)})

def extract_questions(transcript: str) -> list:
    return build_questions_chain(transcript).split("\n")

def build_discussion_points_chain(transcript: str) -> str:
    chain = build_chain("""
        You are an expert video content analyst.
        Your job is to extract the key discussion points from this video transcript.
        These are the core ideas, arguments, or themes the creator spends time explaining or analyzing.
        This includes:
          - Central arguments or positions the creator makes
          - Concepts, frameworks, or models explained in the video
          - Notable examples, comparisons, or case studies brought up
          - Contrasting viewpoints or tensions explored
        Do NOT include minor details, tangents, or filler content.
        Format your reply as a numbered list (e.g. 1. point, 2. point), one point per line.
        If there are no key discussion points, respond with: "no key discussion points found"
    """)
    return chain.invoke({"text": sample_transcript(transcript)})

def extract_key_discussion_points(transcript: str) -> list:
    return build_discussion_points_chain(transcript).split("\n")