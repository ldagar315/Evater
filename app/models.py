from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any

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
