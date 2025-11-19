import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client, Client
from groq import Groq
import dspy
from .dspy_modules import ocr_text

load_dotenv()
logger = logging.getLogger(__name__)

# Initialize Supabase
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_API_KEY"))

# Initialize Groq Client (Global to avoid re-init)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def answer_ocr_extraction(image__url_list: List[str]):
    new_image_url_list = [dspy.Image.from_url(url) for url in image__url_list]
    # Using a specific LM for OCR as per original code
    with dspy.context(lm = dspy.LM('gemini/gemini-2.5-flash', api_key=os.getenv("GEMINI_API_KEY"))):
        ocr_text_answer = ocr_text(answer_sheet_images = new_image_url_list)
    return ocr_text_answer.answer_sheet_text

def merge_qaf(
    questions_list: List[Dict[str, Any]],
    answer_list: List[Dict[str, Any]],
    feedback_list_new: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Build lookup maps by question_number
    answer_map = { ans["question_number"]: ans for ans in answer_list }
    feedback_map = { fb["question_number"]: fb for fb in feedback_list_new }

    merged = []
    for q in questions_list:
        qnum = q["question_number"]
        merged_item = {
            **q,  # all question fields
            "answer": answer_map.get(qnum),
            "feedback": feedback_map.get(qnum),
        }
        merged.append(merged_item)

    return merged

def get_chapter_summary(chapter_name: str, grade:int, subject: str):
    response = (
    supabase.table("Chapter_contents")
    .select("summary")
    .eq("grade", grade)
    .eq("subject", subject)
    .eq("chapter", chapter_name)
    .execute()
    )
    if response.data:
        return response.data[0]["summary"]
    return ""

def get_chapter_structured_summary(chapter_name: str, grade:int, subject: str):
    response = (
    supabase.table("Chapter_contents")
    .select("structured_summary")
    .eq("grade", grade)
    .eq("subject", subject)
    .eq("chapter", chapter_name)
    .execute()
    )
    if response.data:
        return response.data[0]["structured_summary"]
    return {}

def maximum_marks(question_type: str) -> int:
    marks_map = {
        "mcq_single": 1,
        "mcq_multi": 2,
        "true_false": 1,
        "short_answer": 2,
        "long_answer": 3
    }
    return marks_map.get(question_type, 0)

def viva_router(error: str, correctness: int, depth: int, clarity: int, turn_count: int):
    if error == "no error" and correctness > 8 and depth > 8 and clarity > 8 and turn_count >= 2:
        return "Move On"
    elif error == "factual" and correctness < 6:
        return "Ask another question focused on the same factual point, but phrase it differently."
    elif error == "procedural" and correctness < 6:
        return "Generate a question that tests the same procedure in a slightly different way"
    elif error == "application" and depth < 6:
        return "Provide a simple real-world example that demonstrates the application of this concept. Then ask a follow-up question about it."
    elif error == "reasoning":
        return "Ask a 'why' or 'how' question about the reasoning behind the previous answer."
    elif error == "conceptual" and depth < 6:
        return "Ask a foundational question that breaks the concept into simpler parts. Include analogies if useful."
    elif error == "communication/articulation" and clarity < 6:
        return "Ask the student to explain the same answer again in clearer or simpler terms."
    elif error == "metacognitive":
        return "Ask the student how they arrived at the answer or what strategy they used."
    else:
        return "Test this concept using a different question/approach"

def transcribe_audio(audio_file_tuple):
    # Use the global client
    transcription = groq_client.audio.transcriptions.create(
                    file=audio_file_tuple,
                    model="whisper-large-v3",
                    response_format="text",  # Requesting simple text output
                )
    return transcription
