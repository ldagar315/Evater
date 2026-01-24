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
    """
    You are a friendly but probing viva examiner.

    GOAL:
    - Ask ONE clear, open-ended question that the student can answer orally in ~30–60 seconds.
    - Focus on the given concept (and subconcepts inside it).
    - Make the student THINK: prefer "why", "how", "explain", "apply", "compare" over plain definitions.
    - Adapt to the student's current level based on previous questions and answers.

    BEHAVIOUR RULES:
    - NEVER ask yes/no or one-word questions.
    - Avoid multi-part questions like "Explain X and Y and also Z". Ask about ONE thing.
    - If the student previously struggled with this concept:
        - Start simpler, break the idea into smaller pieces, or ask them to explain in their own words.
    - If the student answered well previously:
        - Ask a deeper question: applications, exceptions, edge cases, or real-life scenarios.
    - Use the `special_instructions` very seriously: if it says "focus on factual detail" or
      "ask an application question", shape your question accordingly.
    - Use `state_till_now` to:
        - Avoid repeating earlier questions.
        - Slightly increase or decrease difficulty based on previous scores and errors.
    - The question should be in simple, age-appropriate language for the student's grade.

    OUTPUT FORMAT:
    - Return only the final question sentence, no explanations, no numbering, no quotes.
    """

    concept: Concept = dspy.InputField(desc = "Concept to be tested; includes short description and key sub-points if available.")
    state_till_now: Optional[List[str]] = dspy.InputField(desc = "Previous viva turns on this concept: list of `Q: ... A: ... [score / error]`. Use this to avoid repetition and adjust difficulty.")
    special_instructions: Optional[str] = dspy.InputField(desc = "Guidance from the evaluation of the last answer. Examples: 'student is confused about definition', 'ask an application-level question', 'go deeper on reasoning', 'ask for an example', 'move difficulty slightly up', etc.")
    question: str = dspy.OutputField(desc = "One open-ended viva question, suitable to be answered orally in ~30–60 seconds.")

class EvaluateVivaAnswer(dspy.Signature):
    """
    You are an examiner evaluating an ORAL viva answer.

    GOAL:
    - Judge how well the student's answer addresses the question.
    - Separate correctness, depth of understanding, and clarity of explanation.
    - Identify the MAIN type of error (if any).
    - Produce a structured score object that the system can use for routing.

    THINKING STEPS (do this in your hidden reasoning, not in the output):
    1. Briefly summarise what the student actually said.
    2. Imagine an ideal, high-quality answer for this question.
    3. Compare the student's answer with the ideal answer:
        - Which key ideas are correct?
        - Which important points are missing or wrong?
        - Is the reasoning sound or confused?
        - Is the explanation organised and clear?
    4. Decide:
        - A correctness score (1–10).
        - A depth score (1–10): how deep is their understanding vs surface-level?
        - A clarity/communication score (1–10): how clearly did they explain?
    5. Choose ONE dominant error_type (if there is an error).

    ERROR TYPE GUIDELINES:
    - 'conceptual'  → core idea misunderstood or mixed up.
    - 'procedural' → steps / method / algorithm wrong or incomplete.
    - 'factual'    → specific fact, term, value, or name is wrong or missing.
    - 'application'→ knows theory but fails to apply it to the given situation.
    - 'reasoning'  → illogical, inconsistent, or circular explanation.
    - 'communication/articulation' → ideas roughly right but poorly organised/explained.
    - 'metacognitive' → student openly confused about what they know/don't know.
    - 'no error'   → answer is essentially correct and clear for their grade.

    OUTPUT:
    - `score` must be a structured object (EvaluationOutput) including at least:
        - correctness (1–10)
        - depth (1–10)
        - clarity (1–10)
        - overall (1–10)  [can be an average or weighted]
    - `error_type` must be ONE of the allowed literals above.

    IMPORTANT:
    - Do NOT include your reasoning in the final output; only fill the fields.
    """

    question: str = dspy.InputField(desc = "The viva question asked to the student.")
    answer: str = dspy.InputField(desc = "The student's spoken answer transcribed to text. May contain fillers like 'um', 'I think'. Ignore those while evaluating.")
    score: EvaluationOutput = dspy.OutputField(desc = "Structured scores (correctness, depth, clarity, overall) on a 1–10 scale.")
    error_type: Optional[Literal["conceptual", "procedural", "factual","application" "reasoning", "communication/articulation", "metacognitive", "no error"]] = dspy.OutputField(desc = "Main error type limiting the quality of this answer, or 'no error' if the answer is strong.")

class VivaFeedback(dspy.Signature):
    """ 
    You are a mentor-teacher analysing the entire viva session.

    GOAL:
    - Identify the student's main strengths and weaknesses across ALL concepts.
    - Point out specific knowledge gaps (which concepts, what exactly is missing).
    - Highlight patterns in error types (e.g., often struggles with application or reasoning).
    - Suggest 2–4 concrete next steps the student can take to improve.

    INPUT:
    - `viva_history` is a list of dicts, each containing at least:
        - question
        - answer
        - score (correctness/depth/clarity/overall)
        - error_type

    OUTPUT STYLE (for the student):
    1. Short positive summary (2–3 lines): what they did well.
    2. "Key Gaps" section:
        - 2–5 bullet points, each: [Concept] – what is missing or confused.
    3. "Patterns in Mistakes":
        - Bullet points like: "Often struggles with applying concepts to real-life examples", etc.
    4. "Next Steps (Action Plan)":
        - 3–5 very concrete actions, e.g.:
          - "Re-watch the video for Concept X, and try to explain it in your own words to a friend."
          - "Practice 3 application questions on Y where you predict what will happen and why."
          - "Record yourself answering a 'why' question for Z and check if your explanation is step-by-step."

    TONE:
    - Encouraging, student-friendly, simple language.
    - No marks-shaming; focus on growth and specific improvements.
    """
    viva_history: List[dict] = dspy.InputField(desc = "Full viva history: list of {question, answer, score, error_type} dicts.")
    feedback: str = dspy.OutputField(desc = "Structured written feedback with strengths, key gaps, patterns in errors, and an action plan.")

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
