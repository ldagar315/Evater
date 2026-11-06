import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, ArrowRight, XCircle, Clock, Award } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card, CardContent } from '../components/ui/Card'
import { Header } from '../components/layout/Header'
import { MathText } from '../components/ui/MathRenderer'
import { useAuthContext } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'
import { apiFetch, externalApiUrl } from '../lib/api'
import { QuestionsCreated, Question } from '../types'

interface MCQOption {
  id: string
  text: string
}

interface MCQQuestion extends Question {
  mcq_options: MCQOption[]
  isMultipleCorrect: boolean
}

interface UserAnswer {
  questionId: number
  selectedOptions: string[]
}

export function TakeTestPage() {
  const { testId } = useParams()
  const navigate = useNavigate()
  const { user } = useAuthContext()
  
  const [test, setTest] = useState<QuestionsCreated | null>(null)
  const [questions, setQuestions] = useState<MCQQuestion[]>([])
  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0)
  const [userAnswers, setUserAnswers] = useState<UserAnswer[]>([])
  const [selectedOptions, setSelectedOptions] = useState<string[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [testCompleted, setTestCompleted] = useState(false)
  const [timeElapsed, setTimeElapsed] = useState(0)
  const [feedbackId, setFeedbackId] = useState<number | null>(null)
  const [savingResults, setSavingResults] = useState(false)
  const [saveFailed, setSaveFailed] = useState(false)

  useEffect(() => {
    if (testId && user) {
      fetchTest()
    }
  }, [testId, user])

  useEffect(() => {
    // Timer for elapsed time - only run if test is not completed
    if (!testCompleted) {
      const timer = setInterval(() => {
        setTimeElapsed(prev => prev + 1)
      }, 1000)

      return () => clearInterval(timer)
    }
  }, [testCompleted])

  const fetchTest = async () => {
    try {
      setLoading(true)
      setError('')

      const { data, error } = await supabase
        .from('Questions_Created')
        .select('*')
        .eq('id', parseInt(testId!))
        .eq('created_by', user!.id)
        .single()

      if (error) {
        if (error.code === 'PGRST116') {
          setError('Test not found or you do not have permission to access it')
        } else {
          throw error
        }
        return
      }

      // Check if test has MCQ questions
      const testQuestions = Array.isArray(data.test) ? data.test : []
      const mcqQuestions = testQuestions.filter((q: Question) => 
        q.question_type === 'mcq_single' || q.question_type === 'mcq_multi'
      )

      if (mcqQuestions.length === 0) {
        setError('This test does not contain MCQ questions')
        return
      }

      console.log('Raw MCQ Questions from DB:', mcqQuestions) // Debug log

      // Convert questions to MCQ format with proper option handling
      const formattedQuestions: MCQQuestion[] = mcqQuestions.map((q: Question) => {
        const isMultipleCorrect = q.question_type === 'mcq_multi'
        
        // Answer keys never reach the client: options are rendered as
        // text-only choices and grading happens after submission.
        if (!q.options || q.options.length === 0) {
          throw new Error(`Question ${q.question_number} has no options`)
        }
        const mcqOptions: MCQOption[] = q.options.map((option, index) => ({
          id: `option_${index}`,
          text: typeof option === 'string' ? option : String(option.text ?? option),
        }))

        return {
          ...q,
          mcq_options: mcqOptions,
          isMultipleCorrect
        }
      })

      console.log('Final formatted questions:', formattedQuestions) // Debug log

      setTest(data)
      setQuestions(formattedQuestions)
      setUserAnswers(formattedQuestions.map(q => ({
        questionId: q.question_number,
        selectedOptions: []
      })))
    } catch (err) {
      console.error('Error fetching test:', err)
      setError('Failed to load test. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleOptionSelect = (optionId: string) => {
    const currentQuestion = questions[currentQuestionIndex]
    
    if (currentQuestion.isMultipleCorrect) {
      // Multiple selection
      setSelectedOptions(prev => 
        prev.includes(optionId) 
          ? prev.filter(id => id !== optionId)
          : [...prev, optionId]
      )
    } else {
      // Single selection
      setSelectedOptions([optionId])
    }
  }

  const handleNext = () => {
    // Record the selection and move on. Correctness is only evaluated by the
    // post-submission feedback flow - no answer keys exist on the client.
    const currentQuestion = questions[currentQuestionIndex]
    const updatedAnswers = [...userAnswers]
    updatedAnswers[currentQuestionIndex] = {
      questionId: currentQuestion.question_number,
      selectedOptions: [...selectedOptions]
    }
    setUserAnswers(updatedAnswers)

    if (currentQuestionIndex < questions.length - 1) {
      setCurrentQuestionIndex(prev => prev + 1)
      setSelectedOptions([])
    } else {
      completeTest()
    }
  }

  const completeTest = () => {
    setTestCompleted(true) // This will stop the timer via useEffect dependency
    saveResults()
  }

  const capitalizeDifficulty = (d: string) => {
    const map: Record<string, string> = { easy: 'Easy', medium: 'Medium', hard: 'Hard' }
    return map[(d || '').toLowerCase()] || d
  }

  // Persist the completed MCQ test: generate feedback via the same API the
  // interactive flow uses, then store a FeedbackTest row so the attempt shows
  // up under Previous feedback. Saving never blocks the completion screen.
  const saveResults = async () => {
    if (!test || !user) return
    setSavingResults(true)
    setSaveFailed(false)
    try {
      const payload = {
        test_id: parseInt(testId!),
        questions: questions.map((q) => ({
          question_text: q.question_text,
          question_type: q.question_type,
          question_number: q.question_number,
          maximum_marks: q.maximum_marks,
          difficulty: capitalizeDifficulty(q.difficulty),
          contains_math_expression: q.contains_math_expression,
          options: q.mcq_options.map((o) => ({ text: o.text })),
        })),
        answers: questions.map((q) => {
          const ans = userAnswers.find((a) => a.questionId === q.question_number)
          const selectedText = (ans?.selectedOptions || [])
            .map((id) => q.mcq_options.find((o) => o.id === id)?.text || id)
            .join(', ')
          return {
            question_number: q.question_number,
            answer_text: selectedText,
            image_url: null,
          }
        }),
      }

      const apiUrl = externalApiUrl('/api/gen_feedback_direct')

      const response = await apiFetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!response.ok) throw new Error('Failed to generate feedback')

      const data = await response.json()

      const { data: feedbackData, error: feedbackError } = await supabase
        .from('FeedbackTest')
        .insert([{ given_by: user.id, for_test: parseInt(testId!), feedback: data.feedback }])
        .select()
        .single()

      if (feedbackError) throw feedbackError
      setFeedbackId(feedbackData.id)
    } catch (err) {
      console.error('Failed to save MCQ test results:', err)
      setSaveFailed(true)
    } finally {
      setSavingResults(false)
    }
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getOptionStyle = (option: MCQOption) => {
    const isSelected = selectedOptions.includes(option.id)
    return isSelected
      ? 'bg-primary-100 border-primary-500 text-primary-900'
      : 'bg-white border-neutral-300 text-neutral-700 hover:bg-neutral-50'
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />
        <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6 lg:px-8 lg:py-14">
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
              <p className="text-neutral-600">Loading test...</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (error || !test) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />
        <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6 lg:px-8 lg:py-14">
          <div className="mb-6">
            <Button
              variant="ghost"
              onClick={() => navigate('/previous-tests')}
              className="flex items-center"
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Tests
            </Button>
          </div>
          
          <Card>
            <CardContent className="p-6 text-center">
              <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <XCircle className="h-8 w-8 text-red-600" />
              </div>
              <h2 className="text-xl font-bold text-dark mb-2">Unable to Load Test</h2>
              <p className="text-neutral-600 mb-4">{error}</p>
              <Button onClick={() => navigate('/previous-tests')}>
                Back to Tests
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  if (testCompleted) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />
        <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6 lg:px-8 lg:py-14">
          <Card className="text-center">
            <CardContent className="p-12">
              <div className="w-20 h-20 bg-primary-500 rounded-full flex items-center justify-center mx-auto mb-6">
                <Award className="h-10 w-10 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-dark mb-4">Test Completed!</h1>
              <p className="text-xl text-neutral-600 mb-8">
                Congratulations on completing the MCQ test
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div className="p-6 bg-secondary-50 rounded-xl border border-secondary-200">
                  <div className="text-3xl font-bold text-secondary-600 mb-2">
                    {userAnswers.filter(a => a.selectedOptions.length > 0).length}/{questions.length}
                  </div>
                  <div className="text-sm text-secondary-800">Questions Answered</div>
                </div>
                <div className="p-6 bg-neutral-50 rounded-xl border border-neutral-200">
                  <div className="text-3xl font-bold text-neutral-600 mb-2">{formatTime(timeElapsed)}</div>
                  <div className="text-sm text-neutral-800">Time Taken</div>
                </div>
              </div>

              {savingResults && (
                <p className="text-sm text-neutral-500 mb-4">Saving your results...</p>
              )}
              {saveFailed && (
                <p className="text-sm text-red-500 mb-4">
                  Your results could not be saved. Your answers are recorded above.
                </p>
              )}
              <div className="flex flex-col justify-center gap-4 sm:flex-row">
                {feedbackId && (
                  <Button onClick={() => navigate(`/view-feedback/${feedbackId}`)}>
                    View Feedback
                  </Button>
                )}
                <Button onClick={() => navigate('/previous-tests')} variant="outline">
                  Back to Tests
                </Button>
                <Button onClick={() => navigate('/create-test')} variant={feedbackId ? 'outline' : 'primary'}>
                  Create New Test
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  const currentQuestion = questions[currentQuestionIndex]
  const progress = ((currentQuestionIndex) / questions.length) * 100

  return (
    <div className="min-h-screen bg-cream">
      <Header />

      <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6 sm:py-10 lg:px-8 lg:py-14">
        {/* Header with progress and stats */}
        <div className="mb-6 sm:mb-8 flex items-center justify-between">
          <Button
            variant="ghost"
            onClick={() => navigate("/previous-tests")}
            className="flex items-center text-red-600 bg-red-50"
          >
            <ArrowLeft className="h-4 w-4" />
            Exit Test
          </Button>
              <div className="bg-primary-50 px-6 py-2 flex items-center text-primary-800">
                <Clock className="h-5 w-5 mr-3 text-primary-600" />
                <div className="text-center">
                  <div className="text-base font-semibold text-primary-700 sm:text-lg">
                    {formatTime(timeElapsed)}
                  </div>
                </div>
              </div>
        </div>

        {/* Progress bar */}
        <div className="mb-8">
          <div className="flex justify-between text-sm text-neutral-600 mt-2 font-bold mb-1 sm:mb-2 sm:text-base">
              <span>{Math.round(progress)}% Complete</span>
          </div>
          <div className="w-full bg-neutral-200 rounded-full h-3 shadow-inner">
            <div
              className="bg-primary-500 h-3 rounded-full transition-all duration-300 shadow-sm"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Question Card */}
        <Card className="mb-8 shadow-lg border-neutral-200">
          <CardContent className="p-4 sm:p-8">
          {/* Question type and marks - larger and more prominent */}
              <div className="mb-4 flex flex-wrap justify-end gap-3 sm:mb-8">
                <span className="bg-secondary-50 text-secondary-700 px-2 py-1 rounded-lg text-sm font-normal sm:text-base">
                  {currentQuestion.isMultipleCorrect
                    ? "Multiple Correct"
                    : "Single Correct"}
                </span>
                <span className="bg-green-50 text-green-700 px-2 py-1 rounded-lg text-sm font-normal sm:text-base">
                  {currentQuestion.maximum_marks} marks
                </span>
              </div>
            <div className="text-center mb-4 sm:mb-8">
              {/* Question header with number and text in one row */}
              <div className="max-w-4xl mx-auto mb-4 sm:mb-8 text-left flex items-start">
                {/* Question text */}
                <div className="w-6 h-6 sm:w-8 sm:h-8 sm:text-lg bg-primary-50 rounded-full mr-2 sm:mr-4 flex items-center justify-center shadow-sm">
                  <span className="text-sm font-bold text-primary-600">
                    {currentQuestion.question_number}
                  </span>
                </div>
                <div className="text-sm sm:text-lg text-dark leading-relaxed">
                  {currentQuestion.contains_math_expression ? (
                    <MathText
                      text={currentQuestion.question_text}
                      className="text-dark"
                    />
                  ) : (
                    <p className="text-dark">{currentQuestion.question_text}</p>
                  )}
                </div>
              </div>
            </div>

            {/* Options */}
            <div className="space-y-4 max-w-3xl mx-auto">
              {currentQuestion.mcq_options.map((option, index) => (
                <button
                  key={option.id}
                  onClick={() => handleOptionSelect(option.id)}
                  className={`min-h-11 w-full rounded-xl border-2 p-3 transition-all duration-200 flex items-center justify-between sm:p-4 ${getOptionStyle(
                    option
                  )} cursor-pointer hover:scale-[1.02] hover:shadow-md`}
                >
                  <div className="flex items-center">
                    <div className="w-6 h-6 rounded-full text-sm sm:text-base bg-neutral-600 text-white flex items-center justify-center mr-4">
                      {String.fromCharCode(65 + index)}
                    </div>
                    <span className="text-left text-sm sm:text-base">
                      {currentQuestion.contains_math_expression ? (
                        <MathText text={option.text} />
                      ) : (
                        option.text
                      )}
                    </span>
                  </div>
                </button>
              ))}
            </div>

          </CardContent>
        </Card>

        {/* Navigation */}
        <div className="flex justify-center">
          <Button
            onClick={handleNext}
            disabled={selectedOptions.length === 0}
            size="lg"
            className="px-8 py-3 text-base sm:text-lg font-medium shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
          >
            {currentQuestionIndex < questions.length - 1 ? (
              <>
                Next Question
                <ArrowRight className="h-5 w-5 ml-2" />
              </>
            ) : (
              "Complete Test"
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}
