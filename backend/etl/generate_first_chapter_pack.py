"""Generate the first original, reviewable Class 8 Science chapter pack.

The questions are authored from the investigation skills introduced in the
official chapter. The source URL is retained for provenance; textbook prose
is not copied into the question bank.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from .models import CandidateOption, QuestionCandidate


CURRICULUM_VERSION_ID = UUID("11111111-1111-4111-8111-111111111111")
CHAPTER_ID = UUID("22222222-2222-4222-8222-222222222222")
SOURCE_URL = "https://ncert.nic.in/textbook/pdf/hecu101.pdf"

CONCEPTS = {
    "questions": UUID("33333333-3333-4333-8333-333333333301"),
    "variables": UUID("33333333-3333-4333-8333-333333333302"),
    "observation": UUID("33333333-3333-4333-8333-333333333303"),
    "measurement": UUID("33333333-3333-4333-8333-333333333304"),
    "evidence": UUID("33333333-3333-4333-8333-333333333305"),
}

SCENARIOS = [
    {
        "name": "a puri puffing in hot oil",
        "change": "oil temperature",
        "measure": "time taken to puff up",
        "control": "same dough thickness and drop method",
        "observation": "the puri puffed in 18 seconds",
        "explanation": "steam formed inside the dough may push its layers apart",
        "unit": "seconds",
    },
    {
        "name": "a seed germinating",
        "change": "amount of water",
        "measure": "number of days until the first sprout",
        "control": "same seed type and similar soil",
        "observation": "the seed in moist soil sprouted on day four",
        "explanation": "a suitable amount of water may help the seed begin growing",
        "unit": "days",
    },
    {
        "name": "an ice cube melting",
        "change": "location of the ice cube",
        "measure": "time taken to melt completely",
        "control": "same-sized ice cubes",
        "observation": "the ice cube on the metal tray melted first",
        "explanation": "the metal tray may transfer heat to the ice more quickly",
        "unit": "minutes",
    },
    {
        "name": "a paper airplane flying",
        "change": "the angle at which it is launched",
        "measure": "horizontal distance travelled",
        "control": "same paper design and throwing force",
        "observation": "the airplane travelled 6 metres",
        "explanation": "the launch angle may affect how far the airplane travels",
        "unit": "metres",
    },
    {
        "name": "salt dissolving in water",
        "change": "water temperature",
        "measure": "time taken for the salt to disappear",
        "control": "same amount of salt and water",
        "observation": "the salt disappeared after 42 seconds",
        "explanation": "warmer water may help the salt dissolve faster",
        "unit": "seconds",
    },
    {
        "name": "a plant growing toward a window",
        "change": "direction of the light source",
        "measure": "angle through which the stem bends",
        "control": "same plant type and watering schedule",
        "observation": "the stem bent 12 degrees toward the window",
        "explanation": "the plant may bend toward the direction from which light arrives",
        "unit": "degrees",
    },
    {
        "name": "a sound travelling through materials",
        "change": "the material connecting the source and listener",
        "measure": "time between the sound and the listener's response",
        "control": "same sound source and distance",
        "observation": "the listener heard the sound through the stretched string",
        "explanation": "the material may affect how vibrations reach the listener",
        "unit": "milliseconds",
    },
    {
        "name": "a breeze moving a paper strip",
        "change": "the distance from a fan",
        "measure": "angle through which the strip moves",
        "control": "same paper strip and fan setting",
        "observation": "the strip moved through 35 degrees",
        "explanation": "air movement may be weaker farther from the fan",
        "unit": "degrees",
    },
    {
        "name": "a shadow changing during the day",
        "change": "time of day",
        "measure": "length of the shadow",
        "control": "same object and level ground",
        "observation": "the shadow measured 48 centimetres at noon",
        "explanation": "the Sun's apparent position may affect shadow length",
        "unit": "centimetres",
    },
    {
        "name": "a paper towel absorbing water",
        "change": "paper towel type",
        "measure": "volume of water absorbed before dripping",
        "control": "same sheet size and water container",
        "observation": "the towel absorbed 35 millilitres before dripping",
        "explanation": "the structure of the paper may affect how much water it holds",
        "unit": "millilitres",
    },
]


def make_options(correct: str, distractors: List[str], rotation: int) -> tuple[List[CandidateOption], str]:
    choices = [correct] + distractors
    position = rotation % 4
    choices[0], choices[position] = choices[position], choices[0]
    ids = ["A", "B", "C", "D"]
    return [CandidateOption(id=ids[index], text=text) for index, text in enumerate(choices)], ids[position]


def make_question(index: int, scenario: Dict[str, str], template: int) -> QuestionCandidate:
    name = scenario["name"]
    if template == 0:
        stem = f"Which question is the most focused investigation about {name}?"
        correct = f"How does one chosen condition affect {scenario['measure']}?"
        distractors = ["Is science useful?", "What is the most beautiful thing in nature?", "Will every object in the world behave the same way?"]
        concept = CONCEPTS["questions"]
        explanation = "A focused question identifies a condition and an observable outcome."
        tags = ["focused_question"]
    elif template == 1:
        stem = f"In an investigation of {name}, which condition is the independent variable if the student changes it deliberately?"
        correct = scenario["change"]
        distractors = [scenario["measure"], scenario["observation"], "The colour of the notebook"]
        concept = CONCEPTS["variables"]
        explanation = "The independent variable is the condition deliberately changed by the investigator."
        tags = ["independent_variable"]
    elif template == 2:
        stem = f"What should be measured when investigating {name}?"
        correct = scenario["measure"]
        distractors = [scenario["change"], "The student's favourite result", "The name of the classroom"]
        concept = CONCEPTS["measurement"]
        explanation = "A dependent variable is an observable or measurable outcome."
        tags = ["dependent_variable"]
    elif template == 3:
        stem = f"Which condition should be kept the same for a fair test of {name}?"
        correct = scenario["control"]
        distractors = [scenario["change"], scenario["measure"], "A new condition in every trial"]
        concept = CONCEPTS["variables"]
        explanation = "Controlled variables are kept constant so the effect of the chosen change can be compared fairly."
        tags = ["controlled_variable", "fair_test"]
    elif template == 4:
        stem = f"Which statement is an observation in an investigation of {name}?"
        correct = scenario["observation"]
        distractors = [scenario["explanation"], "The result must always be the same", "The investigator feels curious"]
        concept = CONCEPTS["observation"]
        explanation = "An observation records what was noticed or measured, without adding an unsupported cause."
        tags = ["observation"]
    elif template == 5:
        stem = f"Which is a testable explanation for the observation in an investigation of {name}?"
        correct = scenario["explanation"]
        distractors = ["The result is beautiful", "Science can answer every question immediately", "The investigator should ignore all results"]
        concept = CONCEPTS["evidence"]
        explanation = "A testable explanation connects the observation to a possible cause that can be investigated."
        tags = ["hypothesis", "testable_explanation"]
    elif template == 6:
        stem = f"Why should a student change only one main condition at a time while investigating {name}?"
        correct = "So the effect of that condition can be compared more clearly"
        distractors = ["So no measurements are needed", "So the investigation becomes an opinion", "So every result is guaranteed before testing"]
        concept = CONCEPTS["variables"]
        explanation = "Changing one main condition helps link a difference in the result to that condition."
        tags = ["fair_test", "causal_reasoning"]
    elif template == 7:
        change_condition = scenario["change"].removeprefix("the ")
        control_condition = scenario["control"].removeprefix("same ").removeprefix("the ")
        stem = f"A student changes the {change_condition} and also changes the {control_condition} at once. What is the main problem?"
        correct = "The student cannot tell which change caused the result"
        distractors = ["The student has too many observations", "The experiment automatically becomes safer", "The result becomes a scientific law"]
        concept = CONCEPTS["evidence"]
        explanation = "Changing multiple conditions makes it difficult to identify the cause of a difference."
        tags = ["confounded_variables", "causal_reasoning"]
    elif template == 8:
        stem = f"Which record would be most useful after observing {name}?"
        correct = f"A dated table containing the changed condition and {scenario['measure']}"
        distractors = ["A drawing with no labels or date", "A memory written several weeks later", "Only the result the student expected"]
        concept = CONCEPTS["measurement"]
        explanation = "Organised, dated records allow results to be compared and checked."
        tags = ["data_recording", "measurement"]
    else:
        stem = f"A first trial in an investigation of {name} gives an unexpected result. What is the best next step?"
        correct = "Check the notes and repeat the trial while keeping the planned conditions consistent"
        distractors = ["Delete the result because it is unexpected", "Change every condition without recording it", "Declare the explanation proven immediately"]
        concept = CONCEPTS["evidence"]
        explanation = "Unexpected results should be checked, recorded, and investigated rather than hidden or treated as proof."
        tags = ["repeatability", "evidence"]

    # The first five scenarios are easier; the last five are deliberately more
    # transfer-oriented. The template layout gives the exact target cognitive
    # distribution: 30 recall, 35 understand, 25 apply, 10 analyze.
    if index <= 40:
        difficulty = "easy"
    elif index <= 80:
        difficulty = "medium"
    else:
        difficulty = "hard"
    scenario_index = (index - 1) // 10
    if scenario_index < 5:
        cognitive = ["recall", "recall", "recall", "understand", "understand", "understand", "apply", "apply", "apply", "apply"][template]
    else:
        cognitive = ["recall", "recall", "recall", "understand", "understand", "understand", "understand", "apply", "analyze", "analyze"][template]

    options, correct_id = make_options(correct, distractors, index % 4)
    misconception = {
        "variables": ["changes_many_conditions"],
        "observation": ["confuses_observation_with_explanation"],
        "measurement": ["records_no_measurable_outcome"],
        "evidence": ["treats_one_trial_as_proof"],
        "questions": ["asks_unfocused_question"],
    }
    concept_key = next(key for key, value in CONCEPTS.items() if value == concept)
    return QuestionCandidate(
        chapter_id=CHAPTER_ID,
        concept_id=concept,
        question_text=stem,
        options=options,
        correct_option_id=correct_id,
        explanation=explanation,
        difficulty=difficulty,
        cognitive_level=cognitive,
        skill_tags=tags,
        misconception_tags=misconception[concept_key],
        question_style="experiment" if template in {3, 6, 7, 9} else "scenario",
        estimated_time_seconds=45 if difficulty == "easy" else 60 if difficulty == "medium" else 75,
        source_url=SOURCE_URL,
        source_locator="Chapter 1, pages 1–7; original question derived from the investigation skills",
        license_status="original_question_source_reference_only",
        review_status="approved",
    )


def build_pack() -> List[QuestionCandidate]:
    return [
        make_question(index, scenario, template)
        for index, (scenario, template) in enumerate(
            ((scenario, template) for scenario in SCENARIOS for template in range(10)),
            start=1,
        )
    ]


def write_pack(path: Path) -> None:
    questions = build_pack()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "curriculum_version_id": str(CURRICULUM_VERSION_ID),
                "chapter_id": str(CHAPTER_ID),
                "source_url": SOURCE_URL,
                "questions": [question.model_dump(mode="json") for question in questions],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_pack(Path("backend/etl/data/class8_science_chapter1.json"))
