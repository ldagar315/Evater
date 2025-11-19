import dspy
import os
from typing import List, Optional, Literal
from dotenv import load_dotenv
from .models import (
    Answer, Feedback, Question, TestStructure, TestStructureMCQ, 
    TestStructureSubjective, Concept, EvaluationOutput, Answer
)

load_dotenv()

# Configure DSPy
dspy.configure(
    lm = dspy.LM('openai/gpt-oss-120b', api_key=os.getenv("CEREBRAS_API_KEY"), api_base='https://api.cerebras.ai/v1', cache=False)
)

# Signatures
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

class EvaluateVivaAnswer(dspy.Signature):
    """ Takes a question and answer and evaluates the answer"""

    question: str = dspy.InputField(desc = "Question to be answered orally by the student")
    answer: str = dspy.InputField(desc = "Answer to the question by the student, it is transcribed from the audio")
    score: EvaluationOutput = dspy.OutputField(desc = "Score of the answer, out of 10")
    error_type: Optional[Literal["conceptual", "procedural", "factual","application" "reasoning", "communication/articulation", "metacognitive", "no error"]] = dspy.OutputField(desc = "Type of error in the answer")

class VivaFeedback(dspy.Signature):
    """ Takes viva history and comtemplates actionable feedback for the student"""
    viva_history: List[dict] = dspy.InputField(desc = "Viva history of the student")
    feedback: str = dspy.OutputField(desc = "What are the knowledge gaps in student's understanding ? and What concepts should the child focus upon")

# Modules
result_distribution = dspy.ChainOfThought(GenerateQuestionDistribution)
result_distribution_mcq = dspy.ChainOfThought(GenerateQuestionDistributionMCQ)
result_distribution_subjective = dspy.ChainOfThought(GenerateQuestionDistributionSubjective)
test_generation = dspy.ChainOfThought(GenerateTest)
feedback_generation = dspy.ChainOfThought(Generate_Feedback)
answer_seperation = dspy.ChainOfThought(AnswerSheet)
ocr_text = dspy.ChainOfThought(AnswerSheetToMarkdown)
generate_viva_question = dspy.ChainOfThought(GenerateVivaQuestion)
evaluate_viva_answer = dspy.ChainOfThought(EvaluateVivaAnswer)
viva_feedback = dspy.ChainOfThought(VivaFeedback)
