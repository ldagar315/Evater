import { corsHeaders } from '../_shared/cors.ts'

interface GenerateFeedbackRequest {
  test_id: string
  questions: any[]
  answer_sheets: string[]
}

interface FeedbackResponse {
  question_number: number
  question_text: string
  question_type: string
  maximum_marks: number
  answer: {
    answer: string
  }
  feedback: {
    max_scored: number
    error_type: string
    explanation: string
    next_step: string
  }
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
    const body: GenerateFeedbackRequest = await req.json()
    
    // Validate input
    if (!body.test_id || !body.questions || !body.answer_sheets) {
      return new Response(
        JSON.stringify({ error: 'Missing required fields' }),
        {
          status: 400,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        }
      )
    }

    // Generate feedback for each question
    const merged: FeedbackResponse[] = body.questions.map((question, index) => {
      const scoredMarks = generateRandomScore(question.maximum_marks)
      const errorType = scoredMarks === question.maximum_marks ? 'none' : getRandomErrorType()
      
      return {
        question_number: question.question_number,
        question_text: question.question_text,
        question_type: question.question_type,
        maximum_marks: question.maximum_marks,
        answer: {
          answer: generateSampleAnswer(question)
        },
        feedback: {
          max_scored: scoredMarks,
          error_type: errorType,
          explanation: generateExplanation(question, errorType, scoredMarks),
          next_step: generateNextStep(question, errorType)
        }
      }
    })

    return new Response(
      JSON.stringify({ merged }),
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

function generateRandomScore(maxMarks: number): number {
  // Generate a score with 70% chance of getting at least 60% marks
  const random = Math.random()
  if (random < 0.7) {
    return Math.floor(Math.random() * (maxMarks * 0.4)) + Math.floor(maxMarks * 0.6)
  } else {
    return Math.floor(Math.random() * maxMarks)
  }
}

function getRandomErrorType(): string {
  const errorTypes = ['conceptual', 'calculation', 'presentation', 'incomplete']
  return errorTypes[Math.floor(Math.random() * errorTypes.length)]
}

function generateSampleAnswer(question: any): string {
  const answers = [
    `Student provided a detailed answer for ${question.question_type}`,
    `Answer shows understanding of the topic`,
    `Response includes relevant examples and explanations`,
    `Student attempted to solve the problem systematically`
  ]
  
  return answers[Math.floor(Math.random() * answers.length)]
}

function generateExplanation(question: any, errorType: string, scoredMarks: number): string {
  if (errorType === 'none') {
    return `Excellent work! Your answer demonstrates a clear understanding of the concept and covers all required points.`
  }
  
  const explanations = {
    conceptual: `Your answer shows some understanding, but there are conceptual gaps that need to be addressed. Focus on the fundamental principles.`,
    calculation: `The approach is correct, but there are calculation errors. Double-check your arithmetic and units.`,
    presentation: `Your understanding is good, but the presentation could be clearer. Organize your thoughts better and use proper formatting.`,
    incomplete: `Your answer is on the right track but incomplete. Make sure to address all parts of the question.`
  }
  
  return explanations[errorType as keyof typeof explanations] || 'Good attempt, but there\'s room for improvement.'
}

function generateNextStep(question: any, errorType: string): string {
  const nextSteps = {
    conceptual: 'Review the fundamental concepts and practice similar problems to strengthen your understanding.',
    calculation: 'Practice more computational problems and always verify your calculations.',
    presentation: 'Work on structuring your answers clearly with proper headings and step-by-step solutions.',
    incomplete: 'Read questions carefully and ensure you address all parts before submitting.',
    none: 'Continue practicing similar problems to maintain this level of understanding.'
  }
  
  return nextSteps[errorType as keyof typeof nextSteps] || 'Keep practicing and reviewing the concepts regularly.'
}