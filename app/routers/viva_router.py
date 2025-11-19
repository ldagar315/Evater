from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import asyncio
from ..models import Iterator, EvaluationOutput
from ..services import (
    get_chapter_structured_summary, transcribe_audio, viva_router as get_next_step
)
from ..dspy_modules import (
    generate_viva_question, evaluate_viva_answer, viva_feedback
)

router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws/viva")
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
            
            if not chapter_summary_structured:
                await websocket.send_json({"error": "Chapter not found"})
                continue

            iterator_list = []
            # Ensure 'concepts' key exists
            concepts = chapter_summary_structured.get("concepts", [])
            for i in concepts:
                # Convert dict to Concept model if needed, or just pass dict if Iterator accepts it
                # Assuming i is a dict that matches Concept structure
                # We need to construct Concept object first if Iterator expects it
                # But Iterator expects 'Concept' type. 
                # Let's assume 'i' is a dictionary compatible with Concept model.
                # If 'i' is just a dict, pydantic might handle it if we pass it to Iterator constructor?
                # No, Iterator expects 'concept' field to be Concept type.
                # We should validate it.
                # However, in original code: temp_iterator = Iterator(concept = i, ...)
                # So 'i' was likely a dict and Pydantic coerced it.
                
                temp_iterator = Iterator(
                    concept = i, 
                    memory = [], 
                    score = EvaluationOutput(correctness=0, depth=0, clarity=0),
                    next_step= "none", 
                    turn_count=1
                )
                iterator_list.append(temp_iterator)
                
            for concept in iterator_list:
                while concept.next_step != "Move On":
                    question = generate_viva_question(
                        concept=concept.concept,
                        state_till_now= concept.memory,
                        special_instructions= concept.next_step
                    )
                    await websocket.send_json({"question": question.question})
                    logger.info("-" * 50)
                    logger.info(question.question)
                    
                    audio_webm_bytes = await websocket.receive_bytes()
                    audio_file_tuple = ("audio.webm", audio_webm_bytes)
                    
                    # Run transcription in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    answer = await loop.run_in_executor(None, transcribe_audio, audio_file_tuple)
                    
                    if answer == "exit":
                        break
                    logger.info(f"Your Answer was: {answer}")
                    
                    evaluation = evaluate_viva_answer(question=question.question, answer = answer)
                    
                    logger.info(f"Scores received: Correctness: {evaluation.score.correctness}, Clarity: {evaluation.score.clarity}, Depth: {evaluation.score.depth}")
                    
                    # FIX: Correctly calculate running average
                    # New Average = ((Old Average * (Count - 1)) + New Value) / Count
                    concept.score.correctness = ((concept.score.correctness * (concept.turn_count - 1)) + evaluation.score.correctness) / concept.turn_count
                    concept.score.depth = ((concept.score.depth * (concept.turn_count - 1)) + evaluation.score.depth) / concept.turn_count
                    concept.score.clarity = ((concept.score.clarity * (concept.turn_count - 1)) + evaluation.score.clarity) / concept.turn_count
                    
                    error_type = evaluation.error_type
                    concept.memory.append({
                        "question": question.question, 
                        "answer": answer, 
                        "score": evaluation.score.model_dump(), 
                        "error_type": error_type
                    })

                    await websocket.send_json({"answer": getattr(evaluation, 'reasoning', '')}) # evaluation might not have reasoning field in signature?
                    # Wait, EvaluateVivaAnswer signature has: score, error_type. No reasoning.
                    # Original code: await websocket.send_json({"answer": evaluation.reasoning})
                    # But EvaluateVivaAnswer signature in original code:
                    # class EvaluateVivaAnswer(dspy.Signature): ... score, error_type ...
                    # dspy.ChainOfThought adds 'reasoning' field automatically!
                    # So it should be there.
                    
                    logger.info(f"Errors and Normalised Scores: Correctness: {concept.score.correctness}, Clarity: {concept.score.clarity}, Depth: {concept.score.depth}, Error: {error_type}")
                    
                    concept.next_step = get_next_step(
                        error_type,
                        concept.score.correctness,
                        concept.score.depth,
                        concept.score.clarity,
                        concept.turn_count
                    )
                    logger.info(f"The instructions for the next step is {concept.next_step}")
                    
                    concept.turn_count += 1
                    if concept.turn_count > 3:
                        break
                        
            temp_memory = []
            for i in iterator_list:
                temp_memory.append(i.memory)
            
            scores_dict = {}
            for i in iterator_list:
                scores_dict[i.concept.concept_name] = i.score.model_dump()
                
            viva_feedback_list = viva_feedback(viva_history=temp_memory)
            viva_feedback_text = viva_feedback_list.feedback
            
            final_feedback = {"scores": scores_dict, "feedback": viva_feedback_text}
            await websocket.send_json({"feedback": final_feedback}) 

    except WebSocketDisconnect:
        logger.info("WebSocket connection closed.")
    except Exception as e:
        logger.error(f"Error in WebSocket: {e}")
        await websocket.close(code=1011, reason="An error occurred")
