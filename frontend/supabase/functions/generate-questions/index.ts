import { corsHeaders } from '../_shared/cors.ts'

interface GenerateQuestionsRequest {
  class: number
  subject: string
  topic: string
  difficulty: 'Easy' | 'Medium' | 'Hard'
  length: 'Short' | 'Long'
  special_instructions: string[]
}

interface Question {
  question_text: string
  question_type: 'short_answer' | 'long_answer' | 'mcq_single' | 'mcq_multi' | 'true_false'
  question_number: number
  maximum_marks: number
  difficulty: 'Easy' | 'Medium' | 'Hard'
  contains_math_expression: boolean
  options?: string[]
}

Deno.serve(async (req: Request) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: corsHeaders,
    })
  }

  try {
    const body: GenerateQuestionsRequest = await req.json()
    
    // Validate input
    if (!body.class || !body.subject || !body.topic || !body.difficulty || !body.length) {
      return new Response(
        JSON.stringify({ error: 'Missing required fields' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      )
    }

    // Generate questions based on parameters
    const questionCount = body.length === 'Short' ? 5 : 10
    const questions: Question[] = []

    // Question type distribution based on special instructions
    let questionTypes: Question['question_type'][] = ['short_answer', 'long_answer', 'mcq_single', 'true_false']
    
    if (body.special_instructions.includes('Only MCQ')) {
      questionTypes = ['mcq_single', 'mcq_multi']
    } else if (body.special_instructions.includes('Only Subjective')) {
      questionTypes = ['short_answer', 'long_answer']
    }

    for (let i = 1; i <= questionCount; i++) {
      const questionType = questionTypes[Math.floor(Math.random() * questionTypes.length)]
      
      // Generate question based on subject and topic
      const question: Question = {
        question_text: generateQuestionText(body.subject, body.topic, body.class, questionType, i),
        question_type: questionType,
        question_number: i,
        maximum_marks: getMarksForQuestionType(questionType),
        difficulty: body.difficulty,
        contains_math_expression: ['Mathematics', 'Physics', 'Chemistry'].includes(body.subject),
        options: ['mcq_single', 'mcq_multi', 'true_false'].includes(questionType) 
          ? generateOptions(questionType, body.subject, body.topic)
          : undefined
      }
      
      questions.push(question)
    }

    return new Response(
      JSON.stringify({ questions }),
      {
        status: 200,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  } catch (error) {
    return new Response(
      JSON.stringify({ error: 'Internal server error' }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    )
  }
})

function generateQuestionText(subject: string, topic: string, classLevel: number, questionType: Question['question_type'], questionNumber: number): string {
  const templates = {
    Mathematics: {
      short_answer: `Calculate the ${topic} for the given problem ${questionNumber}.`,
      long_answer: `Explain the concept of ${topic} and solve a related problem with detailed steps.`,
      mcq_single: `Which of the following best represents ${topic}?`,
      mcq_multi: `Select all correct statements about ${topic}:`,
      true_false: `The principle of ${topic} is fundamental in Class ${classLevel} mathematics.`
    },
    Science: {
      short_answer: `Define ${topic} and give one example.`,
      long_answer: `Explain the process of ${topic} with diagrams and examples.`,
      mcq_single: `What is the main characteristic of ${topic}?`,
      mcq_multi: `Which of the following are properties of ${topic}?`,
      true_false: `${topic} is an important concept in ${subject}.`
    },
    default: {
      short_answer: `Briefly explain ${topic}.`,
      long_answer: `Write a detailed essay on ${topic}.`,
      mcq_single: `What is ${topic}?`,
      mcq_multi: `Select all that apply to ${topic}:`,
      true_false: `${topic} is covered in Class ${classLevel} ${subject}.`
    }
  }

  const subjectTemplates = templates[subject as keyof typeof templates] || templates.default
  return subjectTemplates[questionType]
}

function getMarksForQuestionType(questionType: Question['question_type']): number {
  switch (questionType) {
    case 'long_answer':
      return 5
    case 'short_answer':
      return 3
    case 'mcq_single':
    case 'mcq_multi':
    case 'true_false':
      return 1
    default:
      return 2
  }
}

function generateOptions(questionType: Question['question_type'], subject: string, topic: string): string[] {
  if (questionType === 'true_false') {
    return ['True', 'False']
  }

  // Generate subject-specific options
  const genericOptions = [
    `Option A about ${topic}`,
    `Option B about ${topic}`,
    `Option C about ${topic}`,
    `Option D about ${topic}`
  ]

  return genericOptions
}