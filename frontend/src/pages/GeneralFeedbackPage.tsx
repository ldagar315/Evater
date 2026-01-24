import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, MessageSquare, Send } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Card, CardContent, CardHeader } from '../components/ui/Card'
import { Header } from '../components/layout/Header'
import { useAuthContext } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'

export function GeneralFeedbackPage() {
  const navigate = useNavigate()
  const { user } = useAuthContext()
  
  const [name, setName] = useState('')
  const [feedback, setFeedback] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setSubmitting(true)

    try {
      // Validate required fields
      if (!name.trim() || !feedback.trim()) {
        throw new Error('Please fill in all fields')
      }

      // Submit feedback to Feedback_table_2
      const { error } = await supabase
        .from('Feedback_table_2')
        .insert([{
          feedback_by: user!.id,
          name: name.trim(),
          feedback: feedback.trim()
        }])

      if (error) throw error

      setSuccess(true)
      setName('')
      setFeedback('')
      
      // Show success message for 2 seconds then navigate back
      setTimeout(() => {
        navigate('/home')
      }, 2000)
    } catch (err) {
      console.error('Error submitting feedback:', err)
      setError(err instanceof Error ? err.message : 'Failed to submit feedback. Please try again.')
    } finally {
      setSubmitting(false)
    }
  }

  if (success) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />
        
        <div className="max-w-2xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <Card>
            <CardContent className="p-8 text-center">
              <div className="flex justify-center mb-4">
                <div className="w-16 h-16 bg-primary-100 rounded-full flex items-center justify-center">
                  <Send className="h-8 w-8 text-primary-600" />
                </div>
              </div>
              <h2 className="text-2xl font-bold text-dark mb-2">
                Thank You!
              </h2>
              <p className="text-neutral-600 mb-4">
                Your feedback has been submitted successfully. We appreciate your input and will use it to improve Evater.
              </p>
              <p className="text-sm text-neutral-500">
                Redirecting you back to home...
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-cream">
      <Header />
      
      <div className="max-w-2xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
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

        <Card>
          <CardHeader>
            <div className="flex items-center mb-2">
              <MessageSquare className="h-6 w-6 text-primary-600 mr-2" />
              <h2 className="text-2xl font-bold text-dark">Share Your Feedback</h2>
            </div>
            <p className="text-sm text-neutral-600">
              Help us improve Evater by sharing your thoughts, suggestions, or reporting any issues you've encountered.
            </p>
          </CardHeader>
          
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <Input
                label="Your Name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                placeholder="Enter your full name"
                className="w-full"
              />
              
              <div className="space-y-1">
                <label className="block text-sm font-medium text-dark">
                  Feedback for Us
                </label>
                <textarea
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  required
                  placeholder="Share your thoughts, suggestions, or report any issues..."
                  rows={6}
                  className="block w-full rounded-lg border border-neutral-300 bg-white px-3 py-2 shadow-sm transition-colors duration-200 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20 hover:border-neutral-400 resize-vertical"
                />
                <p className="text-xs text-neutral-500">
                  Please be as detailed as possible to help us understand your feedback better.
                </p>
              </div>
              
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <p className="text-sm text-red-600">{error}</p>
                </div>
              )}
              
              <div className="flex space-x-4">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => navigate('/home')}
                  className="flex-1"
                >
                  Cancel
                </Button>
                <Button
                  type="submit"
                  className="flex-1 flex items-center justify-center"
                  loading={submitting}
                  disabled={!name.trim() || !feedback.trim()}
                >
                  {submitting ? (
                    'Submitting...'
                  ) : (
                    <>
                      <Send className="h-4 w-4 mr-2" />
                      Submit Feedback
                    </>
                  )}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Feedback Guidelines */}
        <Card className="mt-6">
          <CardContent className="p-6">
            <h3 className="text-lg font-semibold text-dark mb-3">
              Feedback Guidelines
            </h3>
            <div className="space-y-2 text-sm text-neutral-600">
              <div className="flex items-start">
                <span className="w-2 h-2 bg-primary-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                <p><strong>Bug Reports:</strong> Please describe what you were doing when the issue occurred and what you expected to happen.</p>
              </div>
              <div className="flex items-start">
                <span className="w-2 h-2 bg-secondary-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                <p><strong>Feature Requests:</strong> Tell us about features you'd like to see and how they would help you.</p>
              </div>
              <div className="flex items-start">
                <span className="w-2 h-2 bg-neutral-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                <p><strong>General Feedback:</strong> Share your overall experience and suggestions for improvement.</p>
              </div>
              <div className="flex items-start">
                <span className="w-2 h-2 bg-pink-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                <p><strong>Usability Issues:</strong> Let us know if something is confusing or difficult to use.</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}