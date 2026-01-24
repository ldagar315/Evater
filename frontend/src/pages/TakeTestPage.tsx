import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, ArrowRight, CheckCircle, XCircle, Clock, Award } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card, CardContent } from '../components/ui/Card'
import { Header } from '../components/layout/Header'
import { MathText } from '../components/ui/MathRenderer'
import { useAuthContext } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'
import { QuestionsCreated, Question } from '../types'

interface MCQOption {
  id: string
  text: string
  is_correct: boolean
}

interface MCQQuestion extends Question {
  mcq_options: MCQOption[]
  isMultipleCorrect: boolean
}

interface UserAnswer {
  questionId: number
  selectedOptions: string[]
  isCorrect?: boolean
  correctOptions?: string[]
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
  const [showAnswer, setShowAnswer] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [testCompleted, setTestCompleted] = useState(false)
  const [score, setScore] = useState(0)
  const [timeElapsed, setTimeElapsed] = useState(0)

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

  const playSound = (isCorrect: boolean) => {
    try {
      const soundFile = isCorrect ? '/yeah-boy-114748.mp3' : '/roblox-death-sound-effect.mp3'
      const audio = new Audio(soundFile)
      audio.volume = 0.5 // Set volume to 50%
      audio.play().catch(error => {
        console.log('Could not play sound:', error)
        // Silently fail - don't break the user experience
      })
    } catch (error) {
      console.log('Sound playback error:', error)
      // Silently fail - don't break the user experience
    }
  }

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
        
        // Process options from the database
        let mcqOptions: MCQOption[] = []
        
        if (q.options && q.options.length > 0) {
          console.log('Processing options for question:', q.question_number, q.options) // Debug log
          
          mcqOptions = q.options.map((option, index) => {
            // Handle both string options and object options
            if (typeof option === 'string') {
              // If it's a string, we need to determine correctness differently
              // For now, we'll use a fallback approach
              return {
                id: `option_${index}`,
                text: option,
                is_correct: index === 0 // Fallback: assume first option is correct
              }
            } else if (typeof option === 'object' && option !== null) {
              // Handle object options with is_correct field
              return {
                id: `option_${index}`,
                text: option.text || option.label || String(option),
                is_correct: Boolean(option.is_correct || option.isCorrect || false)
              }
            } else {
              // Fallback for any other format
              return {
                id: `option_${index}`,
                text: String(option),
                is_correct: false
              }
            }
          })
        } else {
          // Generate dummy options if none exist (fallback)
          mcqOptions = generateDummyOptions(q.question_text, isMultipleCorrect)
        }

        console.log('Formatted options for question:', q.question_number, mcqOptions) // Debug log

        // Ensure at least one option is correct
        const hasCorrectOption = mcqOptions.some(opt => opt.is_correct)
        if (!hasCorrectOption && mcqOptions.length > 0) {
          mcqOptions[0].is_correct = true // Make first option correct as fallback
        }

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

  const generateDummyOptions = (questionText: string, isMultipleCorrect: boolean): MCQOption[] => {
    // Generate contextual options based on question content
    const baseOptions = [
      'Option A - Correct answer',
      'Option B - Incorrect answer',
      'Option C - Incorrect answer',
      'Option D - Incorrect answer'
    ]

    if (isMultipleCorrect) {
      return baseOptions.map((text, index) => ({
        id: `option_${index}`,
        text,
        is_correct: index === 0 || index === 2 // Multiple correct for demo
      }))
    } else {
      return baseOptions.map((text, index) => ({
        id: `option_${index}`,
        text,
        is_correct: index === 0 // Single correct
      }))
    }
  }

  const handleOptionSelect = (optionId: string) => {
    if (showAnswer) return // Don't allow selection after showing answer

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
    if (!showAnswer) {
      // Show answer first
      const currentQuestion = questions[currentQuestionIndex]
      const correctOptions = currentQuestion.mcq_options
        .filter(option => option.is_correct) // Use is_correct field
        .map(option => option.id)
      
      console.log('Checking answer for question:', currentQuestion.question_number)
      console.log('Selected options:', selectedOptions)
      console.log('Correct options:', correctOptions)
      
      // Check if answer is correct
      const isCorrect = selectedOptions.length === correctOptions.length &&
        selectedOptions.every(id => correctOptions.includes(id)) &&
        correctOptions.every(id => selectedOptions.includes(id))

      console.log('Is answer correct?', isCorrect)

      // Play sound effect based on correctness
      playSound(isCorrect)

      // Update user answers
      const updatedAnswers = [...userAnswers]
      updatedAnswers[currentQuestionIndex] = {
        questionId: currentQuestion.question_number,
        selectedOptions: [...selectedOptions],
        isCorrect,
        correctOptions
      }
      setUserAnswers(updatedAnswers)
      setShowAnswer(true)
    } else {
      // Move to next question or complete test
      if (currentQuestionIndex < questions.length - 1) {
        setCurrentQuestionIndex(prev => prev + 1)
        setSelectedOptions([])
        setShowAnswer(false)
      } else {
        completeTest()
      }
    }
  }

  const completeTest = () => {
    const correctAnswers = userAnswers.filter(answer => answer.isCorrect).length
    const finalScore = Math.round((correctAnswers / questions.length) * 100)
    setScore(finalScore)
    setTestCompleted(true) // This will stop the timer via useEffect dependency
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getOptionStyle = (option: MCQOption) => {
    const isSelected = selectedOptions.includes(option.id)
    
    if (!showAnswer) {
      return isSelected 
        ? 'bg-primary-100 border-primary-500 text-primary-900' 
        : 'bg-white border-neutral-300 text-neutral-700 hover:bg-neutral-50'
    }

    // Show answer state - use is_correct field
    if (option.is_correct) {
      return 'bg-green-100 border-green-500 text-green-900'
    } else if (isSelected && !option.is_correct) {
      return 'bg-red-100 border-red-500 text-red-900'
    } else {
      return 'bg-neutral-100 border-neutral-300 text-neutral-600'
    }
  }

  const getOptionIcon = (option: MCQOption) => {
    const isSelected = selectedOptions.includes(option.id)
    
    if (!showAnswer) {
      return null
    }

    // Use is_correct field for icon display
    if (option.is_correct) {
      return <CheckCircle className="h-5 w-5 text-green-600" />
    } else if (isSelected && !option.is_correct) {
      return <XCircle className="h-5 w-5 text-red-600" />
    }
    
    return null
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />
        <div className="max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
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
        <div className="max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
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
        <div className="max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <Card className="text-center">
            <CardContent className="p-12">
              <div className="w-20 h-20 bg-primary-500 rounded-full flex items-center justify-center mx-auto mb-6">
                <Award className="h-10 w-10 text-white" />
              </div>
              <h1 className="text-3xl font-bold text-dark mb-4">Test Completed!</h1>
              <p className="text-xl text-neutral-600 mb-8">
                Congratulations on completing the MCQ test
              </p>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                <div className="p-6 bg-primary-50 rounded-xl border border-primary-200">
                  <div className="text-3xl font-bold text-primary-600 mb-2">{score}%</div>
                  <div className="text-sm text-primary-800">Final Score</div>
                </div>
                <div className="p-6 bg-secondary-50 rounded-xl border border-secondary-200">
                  <div className="text-3xl font-bold text-secondary-600 mb-2">
                    {userAnswers.filter(a => a.isCorrect).length}/{questions.length}
                  </div>
                  <div className="text-sm text-secondary-800">Correct Answers</div>
                </div>
                <div className="p-6 bg-neutral-50 rounded-xl border border-neutral-200">
                  <div className="text-3xl font-bold text-neutral-600 mb-2">{formatTime(timeElapsed)}</div>
                  <div className="text-sm text-neutral-800">Time Taken</div>
                </div>
              </div>

              <div className="flex space-x-4 justify-center">
                <Button onClick={() => navigate('/previous-tests')} variant="outline">
                  Back to Tests
                </Button>
                <Button onClick={() => navigate('/create-test')}>
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

      <div className="max-w-4xl mx-auto py-6 sm:py-8 px-4 sm:px-6 lg:px-8">
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
              <div className="flex justify-end space-x-6 mb-4 sm:mb-8">
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
                  disabled={showAnswer}
                  className={`w-full p-2 sm:p-4 rounded-xl border-2 transition-all duration-200 flex items-center justify-between ${getOptionStyle(
                    option
                  )} ${
                    showAnswer
                      ? "cursor-default"
                      : "cursor-pointer hover:scale-[1.02] hover:shadow-md"
                  }`}
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
                  {getOptionIcon(option)}
                </button>
              ))}
            </div>

            {/* Answer explanation */}
            {showAnswer && (
              <div className="mt-8 p-6 bg-neutral-50 rounded-xl max-w-3xl mx-auto border border-neutral-200">
                <h4 className="font-semibold text-dark mb-2">Explanation:</h4>
                <p className="text-neutral-700 text-sm">
                  {userAnswers[currentQuestionIndex]?.isCorrect
                    ? "Correct! Well done on selecting the right answer(s)."
                    : "Incorrect. The correct answer(s) are highlighted in green above."}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Navigation */}
        <div className="flex justify-center">
          <Button
            onClick={handleNext}
            disabled={selectedOptions.length === 0 && !showAnswer}
            size="lg"
            className="px-8 py-3 text-base sm:text-lg font-medium shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
          >
            {!showAnswer ? (
              "Submit Answer"
            ) : currentQuestionIndex < questions.length - 1 ? (
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