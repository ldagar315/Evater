# Evater Backend

This repository contains the backend services for **Evater**, a personalized learning platform designed to generate tests, evaluate answers, and provide detailed feedback for middle school students.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [API Endpoints](#api-endpoints)
- [Setup and Installation](#setup-and-installation)
- [Deployment](#deployment)

---

## Project Overview

Evater aims to revolutionize traditional evaluation methods by offering personalized tests and instant, comprehensive feedback. This backend serves as the core intelligence, handling:

- Test generation  
- Answer sheet processing (via OCR)  
- AI-driven feedback generation  

It's built to be scalable and leverages modern AI models for its core functionalities.

---

## Features

- **Personalized Test Generation**: Creates custom tests based on subject, topic, grade, difficulty level, length, and specific instructions.
- **AI-Powered Question Distribution**: Dynamically determines the optimal mix of question types (MCQ, True/False, Short Answer, Long Answer) using Bloom's Taxonomy.
- **Answer Sheet OCR**: Converts handwritten answer sheets (from images) into machine-readable text.
- **Automated Answer Separation**: Intelligently separates individual answers from a complete answer sheet.
- **Comprehensive Feedback Generation**: Provides detailed feedback for each answer, including explanations, max scores, error types (conceptual, procedural, careless), and next steps.
- **Supabase Integration**: Fetches chapter summaries to ensure tests are generated from specified content.
- **Scalable Deployment**: Designed for serverless deployment using Modal.

---

## Technologies Used

- **Python**: Primary language
- **FastAPI**: Web framework for building APIs
- **DSPy**: Programming model for robust LLM apps
- **Google Gemini (2.5 Flash)**: Used for OCR
- **Supabase**: Database for chapter content
- **Modal**: Serverless deployment
- **Pydantic**: Data validation
- **python-dotenv**: Environment variable management
- **FastAPI-CORS**: CORS handling

---

## Project Structure

- **Environment Variables**: Managed via `.env`  
- **DSPy Configuration**: Sets up LLMs for various tasks  
- **Data Models (Pydantic)**: Defines objects like `Question`, `Answer`, `Feedback`, etc.  
- **DSPy Modules**: Defines DSPy chains like `GenerateTest`, `Generate_Feedback`  
- **Helper Functions**: For processing, database, and marking  
- **FastAPI App**: API endpoints and logic  
- **Modal Config**: Deployment logic for Modal

---

## How It Works

### **Test Generation Flow** (`/api/gen_question`):
1. Receives input like grade, subject, topic, difficulty, length, instructions.
2. Uses `result_distribution` to decide question types.
3. Fetches chapter summary from Supabase.
4. `test_generation` module uses `gpt4.1`.
5. Returns questions to frontend with metadata (e.g., LaTeX, options, marks).

### **Answer Evaluation Flow** (`/api/gen_answer`):
1. Accepts image URLs of answer sheets + original questions.
2. Uses `gemini-2.5-flash` for OCR to extract text.
3. `answer_seperation` extracts individual answers and maps to questions.
4. `feedback_generation` analyzes answers and generates:
   - Explanation
   - Max score
   - Error type
   - Next steps
5. Merges original question, answer, and feedback into a complete structure.

---

## API Endpoints

### `/` (GET)
- **Description**: Root endpoint to check API status.
- **Response**:  
  ```json
  { "message": "Welcome to the Evater API" }
  ```

### `/api/gen_question` (POST)

* **Description**: Generates test based on parameters.
* **Request Body** (`InputDataQuestion`):

  ```json
  {
    "grade": "str",
    "subject": "str",
    "topic": "str",
    "difficulty_level": "Easy | Medium | Hard",
    "length": "Short | Long",
    "special_instructions": ["mcq only", "numerical based only"]
  }
  ```
* **Response**:

  ```json
  {
    "questions": [QuestionObject]
  }
  ```

### `/api/gen_answer` (POST)

* **Description**: Processes answer sheet images and generates feedback.
* **Request Body** (`InputDataAnswer`):

  ```json
  {
    "image_url": ["str"],
    "questions": { "question_id": { ... } }
  }
  ```
* **Response**:

  ```json
  {
    "merged": [MergedQuestionAnswerFeedbackObject]
  }
  ```

### Error Responses

* `400 Bad Request`: Missing input fields
* `500 Internal Server Error`: Server-side error

---

## Setup and Installation

To set up locally:

### Clone the repository:

```bash
git clone <your-repo-url>
cd evater-backend
```

### Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
# Or install manually:
# pip install fastapi uvicorn groq dspy python-dotenv pydantic fastapi-cors supabase google-genai
```

### Create `.env` file:

Add your keys:

```env
SUPABASE_URL="your_supabase_url"
SUPABASE_API_KEY="your_supabase_anon_key"
CEREBRAS_API_KEY="your_cerebras_api_key"
GEMINI_API_KEY="your_google_gemini_api_key"
```

Ensure access to:

* `openai/llama-3.3-70b` (via Cerebras/Groq)
* `gemini/gemini-2.5-flash`


## Deployment

Evater backend is serverless-ready via **Modal**.

* Uses `modal.App`, `@app.function`, and `asgi_app` for deployment config.
* Ensure Modal secrets (`groq-secret`) are set with your API keys.

To deploy:
Follow Modal’s deployment documentation and run relevant deployment commands.

