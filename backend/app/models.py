from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any
from uuid import UUID

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
    test_type: Literal["objective", "subjective", "mixed"]
    special_instructions: List[str]

class InputDataAnswer(BaseModel):
    image_url: Optional[List[str]] = Field(default_factory=list)
    questions: Dict[str, Any]
    text_answers: Optional[Dict[int, str]] = None

class AnswerInput(BaseModel):
    question_number: int
    answer_text: Optional[str] = None
    image_url: Optional[str] = None

class DirectFeedbackRequest(BaseModel):
    questions: List[Question]
    answers: List[AnswerInput]


class QuestionBankTestRequest(BaseModel):
    chapter_id: UUID
    mode: Literal["diagnostic", "practice", "remedial", "challenge"] = "practice"
    question_count: int = Field(default=10, ge=1, le=100)
    block_size: int = Field(default=5, ge=1, le=20)


class QuestionBankAnswer(BaseModel):
    question_id: UUID
    selected_option_id: Optional[Literal["A", "B", "C", "D"]] = None
    answer: Any = None
    response_time_ms: Optional[int] = Field(default=None, ge=0)


class QuestionBankBlockSubmission(BaseModel):
    answers: List[QuestionBankAnswer]


class PublicQuestionOption(BaseModel):
    id: str
    text: str
    media: List[Dict[str, Any]] = Field(default_factory=list)


QuestionBankType = Literal[
    "mcq_single",
    "mcq_multi",
    "assertion_reason",
    "true_false",
    "fill_blank",
    "numerical",
    "case_study",
    "diagram_based",
    "matching",
]


class PublicBankQuestion(BaseModel):
    id: UUID
    concept_id: UUID
    question_type: QuestionBankType
    question_text: str
    options: List[PublicQuestionOption] = Field(default_factory=list)
    media: List[Dict[str, Any]] = Field(default_factory=list)
    render_config: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[str] = None
    hint: Optional[str] = None
    difficulty: Literal["easy", "medium", "hard"]
    cognitive_level: Literal["recall", "understand", "apply", "analyze"]
    skill_tags: List[str] = Field(default_factory=list)
    misconception_tags: List[str] = Field(default_factory=list)
    question_style: Literal["direct", "scenario", "experiment", "data", "diagram"]
    estimated_time_seconds: int
    maximum_marks: int


class RoutingDecision(BaseModel):
    difficulty: Literal["easy", "medium", "hard"]
    focus_concept_ids: List[UUID] = Field(default_factory=list)
    status: Literal["needs_review", "on_track", "challenge_next"] = "on_track"


class QuestionBankTestResponse(BaseModel):
    test_id: UUID
    block_number: int
    block_size: int
    question_count: int
    questions: List[PublicBankQuestion]
    routing: RoutingDecision


class QuestionFlagResponse(BaseModel):
    question_id: UUID
    flagged: bool = True


class QuestionBankAnswerResult(BaseModel):
    question_id: UUID
    is_correct: bool
    marks_awarded: int
    maximum_marks: int
    explanation: Optional[str] = None
    selected_option_id: Optional[Literal["A", "B", "C", "D"]] = None
    correct_option_id: Optional[Literal["A", "B", "C", "D"]] = None
    selected_option_ids: List[str] = Field(default_factory=list)
    correct_option_ids: List[str] = Field(default_factory=list)
    answer_summary: Optional[str] = None
    correct_answer_summary: Optional[str] = None


class QuestionBankBlockResponse(BaseModel):
    test_id: UUID
    block_number: int
    block_score: int
    block_total: int
    percentage: float
    results: List[QuestionBankAnswerResult]
    mastery_updates: List[Dict[str, Any]] = Field(default_factory=list)
    next_block: Optional[QuestionBankTestResponse] = None
    completed: bool = False


class LeaderboardEntry(BaseModel):
    rank: int
    display_name: str
    score: int
    correct_answers: int
    completed_tests: int
    is_current_user: bool = False


class LeaderboardResponse(BaseModel):
    scope: Literal["classroom", "school"]
    scope_label: str
    period: Literal["weekly", "all_time"]
    period_label: str
    scope_available: bool
    membership_message: Optional[str] = None
    entries: List[LeaderboardEntry] = Field(default_factory=list)
    current_user_rank: Optional[int] = None
