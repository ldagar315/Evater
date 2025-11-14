import json
import os
import logging
from typing import Dict, Any, List, Optional, Literal
from fastapi import FastAPI, HTTPException, Depends, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from groq import Groq
from modal import Image, App, Secret, asgi_app
from supabase import create_client, Client
import dspy
import json
load_dotenv()

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_API_KEY"))

################################################################################################
# Defining LM for Dspy #
dspy.configure(
lm = dspy.LM('openai/gpt-oss-120b', api_key=os.getenv("CEREBRAS_API_KEY"), api_base='https://api.cerebras.ai/v1')
)

################################################################################################
# Defining Data Types for Dspy #
class MCQOption(BaseModel):
    text: str
    is_correct: bool


class TestStructure(BaseModel):
    mcq_single_count: int = 0
    mcq_multi_count: int = 0
    true_false_count: int = 0
    short_answer_count: int = 0
    long_answer_count: int = 0

class Answer(BaseModel):
    answer: str
    question_number: int

class Feedback(BaseModel):
    question_number: int
    explanation: str
    max_scored: int
    error_type: Literal["conceptual", "procedural", "careless", "No mistake"] = None
    next_step: str

class Question(BaseModel):
    question_text: str
    question_type: Literal["mcq_single", "mcq_multi", "true_false", "short_answer", "long_answer"]
    difficulty: Literal["Easy", "Medium", "Hard"]
    question_number: int
    options: Optional[List[MCQOption]] = None
    contains_math_expression: bool 
    maximum_marks: int = None

class ErrorResponse(BaseModel):
    error: str

################################################################################################
# Defining Signatures for Dspy #
class AnswerSheet(dspy.Signature):
    answer_sheet_text: str = dspy.InputField()
    answers: List[Answer] = dspy.OutputField()

class Generate_Feedback(dspy.Signature):
    question: Question = dspy.InputField()
    answer: Answer = dspy.InputField()
    feedback: Feedback = dspy.OutputField()

class GenerateQuestionDistribution(dspy.Signature):
    """ Generate a distribution of questions for the test
    Remember to follow Bloom's Taxonomy while generating test
    Don't generate more than 3 questions of the same type
    Minimum questions: 5
    Maximum questions: 15 """
    difficulty_level: Literal["Easy", "Medium", "Hard"] = dspy.InputField()
    length: Literal["Short", "Long"] = dspy.InputField()
    subject: str = dspy.InputField()
    special_instructions: List[str] = dspy.InputField(desc = "Special instructions for the test, like mcq only, short answer only, numerical based only, theory focussed etc.")
    test_structure: TestStructure = dspy.OutputField()

class GenerateTest(dspy.Signature):
    """ Generate a test 
    Format all mathematical content using LaTeX notation wrapped in \( \) delimiters:
    •⁠  ⁠Variables: \(x\), \(y\), \(z\)
    •⁠  ⁠Equations: \(ax^2 + bx + c = 0\)
    •⁠  ⁠Fractions: \(\frac{a}{b}\)
    •⁠  ⁠Exponents: \(x^n\)
    •⁠  ⁠Greek letters: \(\alpha\), \(\beta\), \(\gamma\)
    """
    topic: str = dspy.InputField(desc = 'Chapter/ topic name for the test scope/syllabus')
    topic_covered: str = dspy.InputField(desc = "Scope / Syllabus of the test, Generate test from this content only strictly, don't assume syllabus and generate test")
    test_structure: TestStructure = dspy.InputField(desc = "No. and type of questions to generate")
    grade: str = dspy.InputField(desc = "Grade of the student for which the test is being generated")
    subject: str = dspy.InputField(desc = "Subject of the test")
    special_instructions: List[str] = dspy.InputField(desc = "Special instructions for the test, like mcq only, short answer only, numerical based only, theory focussed etc.")
    test: List[Question] = dspy.OutputField(desc = "Entire Test as a list of questions")

class AnswerSheetToMarkdown(dspy.Signature):
    """Convert hand written answer sheets to Markdown, 
        Most important: 
        - Carefully extract the question number and associated answer
        - Try to convert the equation into latex so that further processing is easy 
        - Don't try to correct any spelling, conceptual or any other kind of mistake, copy the text as it is"""
    answer_sheet_images : List[dspy.Image] =  dspy.InputField(desc = "Images of the answer sheet")
    answer_sheet_text : str = dspy.OutputField(desc = "Text of the answer sheet in Markdown format")

################################################################################################
# Defining Modules for Dspy #
result_distribution = dspy.ChainOfThought(GenerateQuestionDistribution)
test_generation = dspy.ChainOfThought(GenerateTest)
feedback_generation = dspy.ChainOfThought(Generate_Feedback)
answer_seperation = dspy.ChainOfThought(AnswerSheet)
ocr_text = dspy.ChainOfThought(AnswerSheetToMarkdown)

################################################################################################
# Defining Important Functions #
def answer_ocr_extraction(image__url_list: List[str]):
    new_image_url_list = [dspy.Image.from_url(url) for url in image__url_list]
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
    .select("*")
    .eq("grade", grade)
    .eq("subject", subject)
    .execute()
    )
    for i in response.data:
        if i["chapter"] == chapter_name:
            return i["summary"]
def maximum_marks(question_type: str):
    if question_type == "mcq_single":
        return 1
    elif question_type == "mcq_multi":
        return 2
    elif question_type == "true_false":
        return 1
    elif question_type == "short_answer":
        return 2
    elif question_type == "long_answer":
        return 3

##############################################################################################
# Fast API starts from here #

from concurrent.futures import ThreadPoolExecutor
import asyncio

executor = ThreadPoolExecutor()
tasks = []

web_app = FastAPI(
    title = "Evater_v1"
)

web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only. In production, replace "*" with your Flutter app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@web_app.get("/")
def root():
    return {"message": "Welcome to the Evater API"}

class InputData(BaseModel):
    var1: str
    var2: int

class InputDataQuestion(BaseModel):
    grade: str
    subject: str
    topic: str
    difficulty_level: str
    length: str
    special_instructions: List[str]

class InputDataAnswer(BaseModel):
    image_url: List[str]
    questions: Dict[str, Any]

@web_app.post("/test/post", response_model=Dict[str, Any], responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def test_post(InputData: InputData):
    var1 = InputData.var1
    var2 = InputData.var2
    print(var1, var2)
    return {"message": "Test Post", "var1": var1, "var2": var2}

@web_app.on_event("shutdown")
async def shutdown_event():
    print("Gracefully shutting down...")
    
    # Cancel all background asyncio tasks
    for task in tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    # Shutdown the thread pool executor
    executor.shutdown(wait=False)
    print("Shutdown complete.")

@web_app.post("/api/gen_question", response_model=Dict[str, Any], responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def generate_questions(InputDataQuestion: InputDataQuestion):
    """ LLM calls X 2
    Database Call X 1 
    Outputs -> A list of questions objects """
    required_fields = ["difficulty_level", "subject", "length", "special_instructions", "topic", "grade"]
    if not all(hasattr(InputDataQuestion, f) for f in required_fields):
        raise HTTPException(
            status_code=400,
            detail=f"Missing required fields: {', '.join(required_fields)}"
        )

    result = result_distribution(
    difficulty_level= InputDataQuestion.difficulty_level,
    subject = InputDataQuestion.subject,
    length = InputDataQuestion.length, 
    special_instructions = InputDataQuestion.special_instructions
    )
    summary_of_key_points = get_chapter_summary(
        chapter_name =  InputDataQuestion.topic,
        grade = InputDataQuestion.grade,
        subject = InputDataQuestion.subject
    )
    generated_test = test_generation(
    topic=InputDataQuestion.topic,
    topic_covered = summary_of_key_points,
    subject = InputDataQuestion.subject,
    grade = InputDataQuestion.grade, 
    test_structure= result,
    special_instructions = InputDataQuestion.special_instructions
    )

    questions =  [q.model_dump() for q in generated_test.test]
    for i in questions:
        i["maximum_marks"] = maximum_marks(i["question_type"])
    return {"questions": questions}

@web_app.post("/api/gen_answer", response_model=Dict[str, Any], responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
def generate_feedback(InputDataAnswer: InputDataAnswer):
    required_fields = ["image_url", "questions"]
    if not all(hasattr(InputDataAnswer, f) for f in required_fields):
        raise HTTPException(
            status_code=400,
            detail=f"Missing required fields: {', '.join(required_fields)}"
        )
    image_url = InputDataAnswer.image_url
    questions = InputDataAnswer.questions

    text = answer_ocr_extraction(image_url)

    seperated_answers = answer_seperation(
        answer_sheet_text = text
    )


    feedback_list = []
    question_object_list = [Question.model_validate(q) for q in questions['questions']]

    question_map = {q.question_number: q for q in question_object_list}


    for answer in seperated_answers.answers:
        question_object_for_the_answer = question_map.get(answer.question_number)

        if question_object_for_the_answer:
            feedback = feedback_generation(question=question_object_for_the_answer, answer=answer.answer)
            feedback_list.append(feedback)
    feedback_list_new = [feedback.feedback.model_dump() for feedback in feedback_list]

    answer_list = [answer.model_dump() for answer in seperated_answers.answers]

    questions_list = questions['questions']

    merged = merge_qaf(
        questions_list = questions_list,
        answer_list = answer_list,
        feedback_list_new = feedback_list_new
    )
    return {"merged" : merged }

########################################################################################################
# Modal deployment code starts here #


image = Image.debian_slim().pip_install("fastapi","uvicorn","groq","dspy","python-dotenv", "pydantic", "fastapi-cors", "supabase", "google-genai")

app = App("Evater_v1", image=image)

@app.function(
    secrets=[Secret.from_name("groq-secret")],
    min_containers=1
)

@asgi_app()
def wrapper():
    return web_app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("wrapper", host="0.0.0.0", port=port, reload=os.environ.get("FASTAPI_RELOAD", "false").lower() == "true")





# Question Extraction -> This would be needed to extract the questions from the image file or question paper or the answer sheet, 
# Inputs -> A image file of the questions paper or answer sheet
# Output -> Questions, Question Number, Maximum Marks, Question Type -> Stored in the Question Class

# Answer Extraction -> This would be needed to extract the answers from the image file or answer sheet
# Inputs -> A image file of the answer sheet
# Output -> Answer, Question Number -> Stored in the questions class

# Grading and Feedback -> This would be needed to grade the answers and provide feedback
# Inputs -> An object of the question class (question, answer, max marks, question type)
# Output -> marks_obtained, feedback -> Stored in the question class

# Question and Answer Mapping -> This would be needed to map the questions and answers using the question number
# Inputs -> Questions, Answers
# Output -> Answer stored in the question class, under question.answer 
