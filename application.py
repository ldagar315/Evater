import json
import os
import logging
from typing import Dict, Any, List, Optional, Literal
from fastapi import FastAPI, HTTPException, Depends, Query, Request, WebSocket, WebSocketDisconnect
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_API_KEY"))

################################################################################################
# Defining LM for Dspy #
dspy.configure(
lm = dspy.LM('openai/gpt-oss-120b', api_key=os.getenv("CEREBRAS_API_KEY"), api_base='https://api.cerebras.ai/v1', cache= False)
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

class TestStructureMCQ(BaseModel):
    mcq_single_count: int = 0
    mcq_multi_count: int = 0

class TestStructureSubjective(BaseModel):
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

class SubConcept(BaseModel):
    sub_concept_name: str
    description: str
    examples: List[str]
    distractor: List[str] = Field(..., description = "List of things that are usually misinterpreted in the context of this concept")

class Concept(BaseModel):
    concept_name: str
    description: str
    sub_concepts: Optional[list[SubConcept]] = None

class Chapter(BaseModel):
    chapter_name: str
    subject: str
    grade: int
    concepts: list[Concept]

class EvaluationOutput(BaseModel):
    correctness: int = Field(..., description="Score from 1-10 on the factual and procedural accuracy of the answer.")
    depth: int = Field(..., description="Score from 1-10 on the conceptual depth and evidence of reasoning.")
    clarity: int = Field(..., description="Score from 1-10 on the clarity, structure, and use of correct terminology.")

class Iterator(BaseModel):
    concept: Concept
    score: EvaluationOutput
    memory: List[dict]
    next_step: str
    turn_count: int

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

class GenerateQuestionDistributionMCQ(dspy.Signature):
    """ Generate a distribution of mcq questions for the test, 
    if difficulty level is easy, generate single correct mcq only
    if difficulty level is hard, keep around 50% singlecorrect and 50% multicorrect mcqs
    """
    difficulty_level: Literal["Easy", "Medium", "Hard"] = dspy.InputField()
    length: Literal["Short", "Long"] = dspy.InputField()
    subject: str = dspy.InputField()
    test_structure: TestStructureMCQ = dspy.OutputField()

class GenerateQuestionDistributionSubjective(dspy.Signature):
    """ Generate a distribution of questions for the test
    Remember to follow Bloom's Taxonomy while generating test
    Minimum questions: 5
    Maximum questions: 15 """
    difficulty_level: Literal["Easy", "Medium", "Hard"] = dspy.InputField()
    length: Literal["Short", "Long"] = dspy.InputField()
    subject: str = dspy.InputField()
    test_structure: TestStructureSubjective = dspy.OutputField()

class GenerateTest(dspy.Signature):
    """ Generate a test.
    Follow Bloom's Taxonomy while generating the test.
    """
    topic: str = dspy.InputField(desc = 'Chapter/ topic name for the test scope/syllabus')
    topic_covered: str = dspy.InputField(desc = "Scope / Syllabus of the test. Generate test questions ONLY from this content. Do not introduce external information or assume prior knowledge outside this content. Ensure comprehensive coverage within this scope.")
    test_structure: TestStructure = dspy.InputField(desc = "No. and type of questions to generate.")
    grade: str = dspy.InputField(desc = "Grade of the student for which the test is being generated (e.g., '6th Grade', '10th Grade'). This influences the complexity of language and concepts.")
    subject: str = dspy.InputField(desc = "Subject of the test (e.g., 'Science', 'Social Science', 'Mathematics').")
    difficulty: str = dspy.InputField(desc = "Overall difficulty level of the test, aligning with Bloom's Taxonomy. Choose from 'Easy' (focus on Remembering, Understanding), 'Medium' (focus on Applying, Analyzing), or 'Hard' (focus on Evaluating, Creating). This directly influences the cognitive level of questions generated.")
    test: List[Question] = dspy.OutputField(desc = "Entire Test as a list of questions, each carefully crafted to match the specified difficulty and Bloom's Taxonomy levels. Each question should be clear, unambiguous, and directly verifiable against the 'topic_covered' content. Ensure variety in question types as per 'test_structure'.")


class AnswerSheetToMarkdown(dspy.Signature):
    """Convert hand written answer sheets to Markdown, 
        Most important: 
        - Carefully extract the question number and associated answer
        - Try to convert the equation into latex so that further processing is easy 
        - Don't try to correct any spelling, conceptual or any other kind of mistake, copy the text as it is"""
    answer_sheet_images : List[dspy.Image] =  dspy.InputField(desc = "Images of the answer sheet")
    answer_sheet_text : str = dspy.OutputField(desc = "Text of the answer sheet in Markdown format")

class GenerateVivaQuestion(dspy.Signature):
    """ Takes a concept and sub-concept and generates Viva Question for the same, that can be answered orally by the student
    The primary goal of the question is to check the understanding of the student on the concept and sub-concept"""

    concept: Concept = dspy.InputField(desc = "Concept to be tested")
    state_till_now: Optional[List[str]] = dspy.InputField(desc = "The questions and their answer till now, use it to plan next question accordingly and don't repeat questions")
    special_instructions: Optional[str] = dspy.InputField(desc = "Instructions based on evaluation of the last answer")
    question: str = dspy.OutputField(desc = "Question to be answered orally by the student")

generate_viva_question = dspy.ChainOfThought(GenerateVivaQuestion)

class EvaluateVivaAnswer(dspy.Signature):
    """ Takes a question and answer and evaluates the answer"""

    question: str = dspy.InputField(desc = "Question to be answered orally by the student")
    answer: str = dspy.InputField(desc = "Answer to the question by the student, it is transcribed from the audio")
    score: EvaluationOutput = dspy.OutputField(desc = "Score of the answer, out of 10")
    error_type: Optional[Literal["conceptual", "procedural", "factual","application" "reasoning", "communication/articulation", "metacognitive", "no error"]] = dspy.OutputField(desc = "Type of error in the answer")

evaluate_viva_answer = dspy.ChainOfThought(EvaluateVivaAnswer)

class VivaFeedback(dspy.Signature):
    """ Takes viva history and comtemplates actionable feedback for the student"""
    viva_history: List[dict] = dspy.InputField(desc = "Viva history of the student")
    feedback: str = dspy.OutputField(desc = "What are the knowledge gaps in student's understanding ? and What concepts should the child focus upon")

viva_feedback = dspy.ChainOfThought(VivaFeedback)

################################################################################################
# Defining Modules for Dspy #
result_distribution = dspy.ChainOfThought(GenerateQuestionDistribution)
result_distribution_mcq = dspy.ChainOfThought(GenerateQuestionDistribution)
result_distribution_subjective = dspy.ChainOfThought(GenerateQuestionDistribution)
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
    .select("summary")
    .eq("grade", grade)
    .eq("subject", subject)
    .eq("chapter", chapter_name)
    .execute()
    )
    return response.data[0]["summary"]

def get_chapter_structured_summary(chapter_name: str, grade:int, subject: str):
    response = (
    supabase.table("Chapter_contents")
    .select("structured_summary")
    .eq("grade", grade)
    .eq("subject", subject)
    .eq("chapter", chapter_name)
    .execute()
    )
    return response.data[0]["structured_summary"]

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
    client_transcribe = Groq(api_key=os.getenv("GROQ_API_KEY"))
    transcription = client_transcribe.audio.transcriptions.create(
                    file=audio_file_tuple,
                    model="whisper-large-v3",
                    response_format="text",  # Requesting simple text output
                )
    return transcription

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

    if "only mcq" in InputDataQuestion.special_instructions:
        result = result_distribution_mcq(
    difficulty_level= InputDataQuestion.difficulty_level,
    subject = InputDataQuestion.subject,
    length = InputDataQuestion.length, 
    special_instructions = InputDataQuestion.special_instructions
    )
    elif "only subjective" in InputDataQuestion.special_instructions:
        result = result_distribution_subjective(
    difficulty_level= InputDataQuestion.difficulty_level,
    subject = InputDataQuestion.subject,
    length = InputDataQuestion.length, 
    special_instructions = InputDataQuestion.special_instructions
    )
    else:
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
    difficulty = InputDataQuestion.difficulty_level
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

@web_app.websocket("/ws/viva")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to receive audio from a client and send back dummy data.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted.")
    try:
        # Send some dummy data upon connection
        await websocket.send_json({"status": "connected", "message": "Ready to receive audio."})

        while True:
            # Receive audio data from the client
            chapter_information = await websocket.receive_json()
            logger.info(f"Received chapter information: {chapter_information}")
            chapter_summary_structured = get_chapter_structured_summary(
                chapter_name = chapter_information["chapter"],
                grade = chapter_information["grade"],
                subject = chapter_information["subject"]
            )
            iterator_list = []
            for i in chapter_summary_structured["concepts"]:
                temp_iterator = Iterator(concept = i, memory = [], score = EvaluationOutput(correctness=0, depth=0, clarity=0),next_step= "none", turn_count=1)
                iterator_list.append(temp_iterator)
            for concept in iterator_list:
                while concept.next_step != "Move On":
                    question = generate_viva_question(concept=concept.concept,state_till_now= concept.memory,special_instructions= concept.next_step)
                    await websocket.send_json({"question": question.question})
                    print("-"*50)
                    print(question.question)
                    
                    audio_webm_bytes = await websocket.receive_bytes()
                    audio_file_tuple = ("audio.webm", audio_webm_bytes)
                    answer = transcribe_audio(audio_file_tuple)
                    if answer == "exit":
                        break
                    print(f"Your Answer was: {answer}")
                    
                    evaluation = evaluate_viva_answer(question=question.question, answer = answer)
                    
                    print(f"Scores recieved in the evaluation are: \n Correctness: {evaluation.score.correctness}, \n Clarity: {evaluation.score.clarity} \n Depth: {evaluation.score.depth} ")
                    concept.score.correctness += evaluation.score.correctness 
                    concept.score.correctness = concept.score.correctness / concept.turn_count
                    concept.score.depth += evaluation.score.depth 
                    concept.score.depth = concept.score.depth / concept.turn_count
                    concept.score.clarity += evaluation.score.clarity 
                    concept.score.clarity = concept.score.clarity / concept.turn_count
                    error_type = evaluation.error_type
                    concept.memory.append({"question": question.question, "answer": answer, "score": evaluation.score.model_dump(), "error_type": error_type})

                    await websocket.send_json({"answer": evaluation.reasoning})
                    print(f"Errors and Normalised Scores are: \n Correctness: {concept.score.correctness}, \n Clarity: {concept.score.clarity} \n Depth: {concept.score.depth} \n Error: {error_type}")
                    
                    concept.next_step = viva_router(error_type,concept.score.correctness,concept.score.depth,concept.score.clarity,concept.turn_count)
                    print(f"The instructions for the next step is {concept.next_step}")
                    
                    concept.turn_count += 1
                    if concept.turn_count > 3:
                        break
            temp_memory = []
            for i in iterator_list:
                temp_memory.append(i.memory)
            # write code for the dict that contains concept name by scores
            scores_dict = {}
            for i in iterator_list:
                scores_dict[i.concept.concept_name] = i.score.model_dump()
            viva_feedback_list = viva_feedback(viva_history=temp_memory)
            viva_feedback_list = viva_feedback_list.feedback
            final_feedback = {"scores": scores_dict, "feedback": viva_feedback_list}
            await websocket.send_json({"feedback": final_feedback}) 

            



    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"Error in WebSocket: {e}")
        await websocket.close(code=1011, reason="An error occurred")



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


