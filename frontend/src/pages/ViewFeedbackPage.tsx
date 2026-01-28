import React, { useEffect, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { ArrowLeft, CheckCircle, XCircle, AlertCircle } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card, CardContent, CardHeader } from '../components/ui/Card'
import { Header } from '../components/layout/Header'
import { supabase } from '../lib/supabase'
import { FeedbackTest, FeedbackResponse } from '../types'

export function ViewFeedbackPage() {
  const { evaluationId } = useParams()
  const navigate = useNavigate()
  const [evaluation, setEvaluation] = useState<FeedbackTest | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (evaluationId) {
      fetchEvaluation()
    } else {
      fetchLatestEvaluation()
    }
  }, [evaluationId])

  const fetchEvaluation = async () => {
    try {
      const { data, error } = await supabase
        .from('FeedbackTest')
        .select('*')
        .eq('id', parseInt(evaluationId!))
        .single()

      if (error) throw error
      setEvaluation(data)
    } catch (err) {
      setError('Failed to load evaluation')
    } finally {
      setLoading(false)
    }
  }

  const fetchLatestEvaluation = async () => {
    try {
      const { data, error } = await supabase
        .from('FeedbackTest')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(1)
        .single()

      if (error) throw error
      setEvaluation(data)
    } catch (err) {
      setError('No evaluations found')
    } finally {
      setLoading(false)
    }
  }

  const getScoreColor = (scored: number, total: number) => {
    const percentage = (scored / total) * 100
    if (percentage >= 80) return 'text-green-600'
    if (percentage >= 60) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getErrorIcon = (errorType: string) => {
    switch (errorType) {
      case 'none':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'conceptual':
        return <XCircle className="h-5 w-5 text-red-500" />
      default:
        return <AlertCircle className="h-5 w-5 text-yellow-500" />
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Header />
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      </div>
    )
  }

  if (error || !evaluation) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Header />
        <div className="max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <Card>
            <CardContent className="p-6 text-center">
              <p className="text-gray-600 mb-4">{error || 'Evaluation not found'}</p>
              <Button onClick={() => navigate('/home')}>
                Back to Home
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  const feedback = Array.isArray(evaluation.feedback) ? evaluation.feedback : []
  const totalMarks = feedback.reduce((sum: number, item: FeedbackResponse) => sum + (item.maximum_marks || 0), 0)
  const scoredMarks = feedback.reduce((sum: number, item: FeedbackResponse) => sum + (item.feedback?.max_scored || 0), 0)
  const percentage = totalMarks > 0 ? Math.round((scoredMarks / totalMarks) * 100) : 0

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <div className="max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="mb-6 flex items-center justify-between">
          <Button
            variant="ghost"
            onClick={() => navigate('/home')}
            className="flex items-center"
          >
            <ArrowLeft className="h-4 w-4 mr-2" />
            Back to Home
          </Button>
        </div>

        {/* Overall Results */}
        <Card className="mb-6">
          <CardHeader>
            <h1 className="text-2xl font-bold text-gray-900">Evaluation Results</h1>
            <p className="text-sm text-gray-600">
              Completed on {new Date(evaluation.created_at).toLocaleDateString()}
            </p>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="text-center">
                <div className={`text-3xl font-bold ${getScoreColor(scoredMarks, totalMarks)}`}>
                  {scoredMarks}/{totalMarks}
                </div>
                <p className="text-sm text-gray-600">Total Score</p>
              </div>
              <div className="text-center">
                <div className={`text-3xl font-bold ${getScoreColor(scoredMarks, totalMarks)}`}>
                  {percentage}%
                </div>
                <p className="text-sm text-gray-600">Percentage</p>
              </div>
              <div className="text-center">
                <div className="text-3xl font-bold text-blue-600">
                  {feedback.length}
                </div>
                <p className="text-sm text-gray-600">Questions</p>
              </div>
            </div>
            
            <div className="mt-6 p-4 bg-blue-50 rounded-lg">
              <h3 className="font-semibold text-blue-900 mb-2">Overall Summary</h3>
              <p className="text-blue-800">
                {percentage >= 80 
                  ? 'Excellent performance! You have a strong understanding of the concepts.'
                  : percentage >= 60
                  ? 'Good work! There are some areas where you can improve.'
                  : 'Keep practicing! Focus on the areas highlighted below to improve your understanding.'
                }
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Question-wise Feedback */}
        <div className="space-y-4">
          <h2 className="text-xl font-bold text-gray-900">Question-wise Feedback</h2>
          
          {feedback.map((item: FeedbackResponse, index: number) => (
            <Card key={index}>
              <CardContent className="p-6">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Question {item.question_number || index + 1}
                  </h3>
                  <div className="flex items-center space-x-2">
                    {getErrorIcon(item.feedback?.error_type || 'unknown')}
                    <span className={`font-semibold ${getScoreColor(item.feedback?.max_scored || 0, item.maximum_marks || 0)}`}>
                      {item.feedback?.max_scored || 0}/{item.maximum_marks || 0}
                    </span>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium text-gray-700 mb-1">Question:</h4>
                    <p className="text-gray-600">{item.question_text || 'No question text available'}</p>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium text-gray-700 mb-1">Your Answer:</h4>
                      <div className="bg-gray-50 p-3 rounded-lg">
                        <p className="text-gray-700">{item.answer?.answer || 'No answer provided'}</p>
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-gray-700 mb-1">Type:</h4>
                      <div className="flex items-center space-x-2">
                        <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">
                          {item.question_type?.replace('_', ' ') || 'Unknown'}
                        </span>
                        {item.feedback?.error_type && item.feedback.error_type !== 'none' && (
                          <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-sm">
                            {item.feedback.error_type} error
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  {item.feedback?.explanation && (
                    <div>
                      <h4 className="font-medium text-gray-700 mb-1">Feedback:</h4>
                      <div className="bg-yellow-50 p-3 rounded-lg">
                        <p className="text-yellow-800">{item.feedback.explanation}</p>
                      </div>
                    </div>
                  )}
                  
                  {item.feedback?.next_step && (
                    <div>
                      <h4 className="font-medium text-gray-700 mb-1">Next Steps:</h4>
                      <div className="bg-green-50 p-3 rounded-lg">
                        <p className="text-green-800">{item.feedback.next_step}</p>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}