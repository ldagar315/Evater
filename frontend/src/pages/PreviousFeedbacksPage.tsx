import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, MessageSquare, Calendar, BookOpen, Eye, Search, Filter, Award, TrendingUp } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Select } from '../components/ui/Select'
import { Card, CardContent, CardHeader } from '../components/ui/Card'
import { Header } from '../components/layout/Header'
import { useAuthContext } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'
import { FeedbackTest, QuestionsCreated, FeedbackResponse } from '../types'

interface FeedbackWithTest extends FeedbackTest {
  test_details?: QuestionsCreated
  marks_scored?: number
  total_marks?: number
  percentage?: number
}

export function PreviousFeedbacksPage() {
  const navigate = useNavigate()
  const { user } = useAuthContext()
  
  const [feedbacks, setFeedbacks] = useState<FeedbackWithTest[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [searchTerm, setSearchTerm] = useState('')
  const [subjectFilter, setSubjectFilter] = useState('')
  const [difficultyFilter, setDifficultyFilter] = useState('')

  useEffect(() => {
    if (user) {
      fetchFeedbacks()
    }
  }, [user])

  const fetchFeedbacks = async () => {
    try {
      setLoading(true)
      setError('')

      // First, fetch all feedbacks for the current user
      const { data: feedbackData, error: feedbackError } = await supabase
        .from('FeedbackTest')
        .select('*')
        .eq('given_by', user!.id)
        .order('created_at', { ascending: false })

      if (feedbackError) throw feedbackError

      if (!feedbackData || feedbackData.length === 0) {
        setFeedbacks([])
        return
      }

      // Get unique test IDs to fetch test details
      const testIds = [...new Set(feedbackData.map(f => f.for_test).filter(Boolean))]
      
      // Fetch test details for all referenced tests
      const { data: testData, error: testError } = await supabase
        .from('Questions_Created')
        .select('*')
        .in('id', testIds)

      if (testError) throw testError

      // Create a map of test details for quick lookup
      const testMap = new Map(testData?.map(test => [test.id, test]) || [])

      // Process feedbacks and calculate scores
      const processedFeedbacks: FeedbackWithTest[] = feedbackData.map(feedback => {
        const testDetails = testMap.get(feedback.for_test!)
        let marksScored = 0
        let totalMarks = 0

        // Calculate marks from feedback data
        if (feedback.feedback && Array.isArray(feedback.feedback)) {
          const feedbackArray = feedback.feedback as FeedbackResponse[]
          marksScored = feedbackArray.reduce((sum, item) => sum + (item.feedback?.max_scored || 0), 0)
          totalMarks = feedbackArray.reduce((sum, item) => sum + (item.maximum_marks || 0), 0)
        }

        const percentage = totalMarks > 0 ? Math.round((marksScored / totalMarks) * 100) : 0

        return {
          ...feedback,
          test_details: testDetails,
          marks_scored: marksScored,
          total_marks: totalMarks,
          percentage
        }
      })

      setFeedbacks(processedFeedbacks)
    } catch (err) {
      console.error('Error fetching feedbacks:', err)
      setError('Failed to load feedback data. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  // Get unique subjects and difficulties for filters
  const uniqueSubjects = [...new Set(
    feedbacks
      .map(feedback => feedback.test_details?.subject)
      .filter(Boolean)
  )].sort()
  
  const uniqueDifficulties = [...new Set(
    feedbacks
      .map(feedback => feedback.test_details?.difficulty_level)
      .filter(Boolean)
  )].sort()

  // Filter feedbacks based on search and filters
  const filteredFeedbacks = feedbacks.filter(feedback => {
    const subject = feedback.test_details?.subject || ''
    const topic = feedback.test_details?.chapter || ''
    const difficulty = feedback.test_details?.difficulty_level || ''

    const matchesSearch = searchTerm === '' || 
      subject.toLowerCase().includes(searchTerm.toLowerCase()) ||
      topic.toLowerCase().includes(searchTerm.toLowerCase())
    
    const matchesSubject = subjectFilter === '' || subject === subjectFilter
    const matchesDifficulty = difficultyFilter === '' || difficulty === difficultyFilter

    return matchesSearch && matchesSubject && matchesDifficulty
  })

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const getPerformanceColor = (percentage: number) => {
    if (percentage >= 90) return 'text-green-600'
    if (percentage >= 80) return 'text-blue-600'
    if (percentage >= 70) return 'text-yellow-600'
    if (percentage >= 60) return 'text-orange-600'
    return 'text-red-600'
  }

  const getPerformanceBadgeColor = (percentage: number) => {
    if (percentage >= 90) return 'bg-green-100 text-green-800'
    if (percentage >= 80) return 'bg-blue-100 text-blue-800'
    if (percentage >= 70) return 'bg-yellow-100 text-yellow-800'
    if (percentage >= 60) return 'bg-orange-100 text-orange-800'
    return 'bg-red-100 text-red-800'
  }

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty.toLowerCase()) {
      case 'easy':
        return 'bg-green-100 text-green-800'
      case 'medium':
        return 'bg-yellow-100 text-yellow-800'
      case 'hard':
        return 'bg-red-100 text-red-800'
      default:
        return 'bg-neutral-100 text-neutral-800'
    }
  }

  const getPerformanceLabel = (percentage: number) => {
    if (percentage >= 90) return 'Excellent'
    if (percentage >= 80) return 'Good'
    if (percentage >= 70) return 'Average'
    if (percentage >= 60) return 'Below Average'
    return 'Needs Improvement'
  }

  // Calculate stats
  const averageScore = filteredFeedbacks.length > 0 
    ? Math.round(filteredFeedbacks.reduce((sum, feedback) => sum + (feedback.percentage || 0), 0) / filteredFeedbacks.length)
    : 0

  const totalMarksScored = filteredFeedbacks.reduce((sum, feedback) => sum + (feedback.marks_scored || 0), 0)
  const totalMarksAvailable = filteredFeedbacks.reduce((sum, feedback) => sum + (feedback.total_marks || 0), 0)

  if (loading) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />
        <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
              <p className="text-neutral-600">Loading your feedback history...</p>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-cream">
      <Header />

      <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="mb-6">
          <div className="flex items-center mb-2">
            <h1 className="text-2xl font-bold text-dark">Previous Feedbacks</h1>
          </div>
          <p className="text-sm text-neutral-600">
            View and analyze all your previous test evaluations and performance
            feedback
          </p>
        </div>

        {error && (
          <Card className="mb-6">
            <CardContent className="p-4">
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Performance Overview */}
        {filteredFeedbacks.length > 0 && (
          <Card className="mb-6">
            <CardContent className="p-6">
              <div className="flex items-center mb-4">
                <TrendingUp className="h-5 w-5 text-primary-600 mr-2" />
                <h3 className="text-lg font-semibold text-dark">
                  Performance Overview
                </h3>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-primary-50 rounded-lg">
                  <div className="text-2xl font-bold text-primary-600">
                    {filteredFeedbacks.length}
                  </div>
                  <div className="text-sm text-primary-800">
                    Total Evaluations
                  </div>
                </div>
                <div className="text-center p-4 bg-secondary-50 rounded-lg">
                  <div
                    className={`text-2xl font-bold ${getPerformanceColor(
                      averageScore
                    )}`}
                  >
                    {averageScore}%
                  </div>
                  <div className="text-sm text-secondary-800">
                    Average Score
                  </div>
                </div>
                </div>
            </CardContent>
          </Card>
        )}

        {/* Performance Insights */}
        {filteredFeedbacks.length > 0 && (
          <Card className="mt-6 mb-6">
            <CardContent className="p-6">
              <h3 className="text-lg font-semibold text-dark mb-4 flex items-center">
                <TrendingUp className="h-5 w-5 text-primary-600 mr-2" />
                Performance Insights
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="p-4 bg-green-50 rounded-lg border border-green-200">
                  <h4 className="font-semibold text-green-900 mb-2">
                    Best Performance
                  </h4>
                  <div className="text-2xl font-bold text-green-600 mb-1">
                    {Math.max(
                      ...filteredFeedbacks.map((f) => f.percentage || 0)
                    )}
                    %
                  </div>
                  <p className="text-sm text-green-700">
                    Your highest score achieved
                  </p>
                </div>

                <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                  <h4 className="font-semibold text-blue-900 mb-2">
                    Most Attempted
                  </h4>
                  <div className="text-lg font-bold text-blue-600 mb-1">
                    {uniqueSubjects.length > 0
                      ? uniqueSubjects.reduce((prev, current) =>
                          filteredFeedbacks.filter(
                            (f) => f.test_details?.subject === current
                          ).length >
                          filteredFeedbacks.filter(
                            (f) => f.test_details?.subject === prev
                          ).length
                            ? current
                            : prev
                        )
                      : "N/A"}
                  </div>
                  <p className="text-sm text-blue-700">
                    Subject with most evaluations
                  </p>
                </div>

                <div className="p-4 bg-purple-50 rounded-lg border border-purple-200">
                  <h4 className="font-semibold text-purple-900 mb-2">
                    Improvement Trend
                  </h4>
                  <div className="text-lg font-bold text-purple-600 mb-1">
                    {filteredFeedbacks.length >= 2 &&
                    (filteredFeedbacks[0].percentage || 0) >
                      (filteredFeedbacks[1].percentage || 0)
                      ? "↗️ Improving"
                      : filteredFeedbacks.length >= 2 &&
                        (filteredFeedbacks[0].percentage || 0) <
                          (filteredFeedbacks[1].percentage || 0)
                      ? "↘️ Declining"
                      : "→ Stable"}
                  </div>
                  <p className="text-sm text-purple-700">
                    Based on recent evaluations
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Filters and Search */}
        <Card className="mb-6">
          <CardContent className="p-6">
            <div className="flex items-center mb-4">
              <Filter className="h-5 w-5 text-neutral-600 mr-2" />
              <h3 className="text-lg font-semibold text-dark">
                Filter Feedbacks
              </h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-neutral-400" />
                <Input
                  placeholder="Search by subject or topic..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10"
                />
              </div>
              <Select
                value={subjectFilter}
                onChange={(e) => setSubjectFilter(e.target.value)}
                options={[
                  { value: "", label: "All Subjects" },
                  ...uniqueSubjects.map((subject) => ({
                    value: subject,
                    label: subject,
                  })),
                ]}
              />
              <Select
                value={difficultyFilter}
                onChange={(e) => setDifficultyFilter(e.target.value)}
                options={[
                  { value: "", label: "All Difficulties" },
                  ...uniqueDifficulties.map((difficulty) => ({
                    value: difficulty,
                    label: difficulty,
                  })),
                ]}
              />
              <Button
                variant="outline"
                onClick={() => {
                  setSearchTerm("");
                  setSubjectFilter("");
                  setDifficultyFilter("");
                }}
                className="flex items-center justify-center"
              >
                Clear Filters
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Results Summary */}
        <div className="mb-6 flex items-center justify-between">
          <p className="text-sm text-neutral-600">
            Showing {filteredFeedbacks.length} of {feedbacks.length} evaluations
          </p>
          <div className="flex items-center space-x-2 text-sm text-neutral-600">
            <Calendar className="h-4 w-4" />
            <span>Sorted by evaluation date (newest first)</span>
          </div>
        </div>
        
        <Card>
          <CardContent className="p-0">
            {filteredFeedbacks.length === 0 ? (
              // --- Empty State (Same for Mobile and Desktop) ---
              <div className="p-12 text-center">
                <MessageSquare className="h-12 w-12 text-neutral-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-dark mb-2">
                  {feedbacks.length === 0
                    ? "No Evaluations Yet"
                    : "No Evaluations Found"}
                </h3>
                <p className="text-neutral-600 mb-4">
                  {feedbacks.length === 0
                    ? "You haven't submitted any answer sheets for evaluation yet. Create a test and submit your answers to get started!"
                    : "No evaluations match your current filters. Try adjusting your search criteria."}
                </p>
                <div className="flex space-x-4 justify-center">
                  <Button
                    onClick={() => navigate("/create-test")}
                    className="flex items-center bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg px-4 py-2 transition duration-150"
                  >
                    <BookOpen className="h-4 w-4 mr-2" />
                    {feedbacks.length === 0
                      ? "Create Your First Test"
                      : "Create New Test"}
                  </Button>
                  {feedbacks.length === 0 && (
                    <Button
                      onClick={() => navigate("/previous-tests")}
                      variant="outline"
                      className="flex items-center border border-neutral-300 text-neutral-700 hover:bg-neutral-50 rounded-lg px-4 py-2 transition duration-150"
                    >
                      <Eye className="h-4 w-4 mr-2" />
                      View Previous Tests
                    </Button>
                  )}
                </div>
              </div>
            ) : (
              <>
                {/* ---------------------------------------------------- */}
                {/* 1. MOBILE/TABLET VIEW (Cards) - Visible on screens < lg */}
                {/* ---------------------------------------------------- */}
                <div className="lg:hidden p-4 space-y-4">
                  {filteredFeedbacks.map((feedback) => {
                    const [datePart, timePart] = formatDate(
                      feedback.created_at
                    ).split(",");
                    const marksScored = feedback.marks_scored || 0;
                    const totalMarks = feedback.total_marks || 0;
                    const percentage = feedback.percentage || 0;

                    return (
                      <div
                        key={feedback.id}
                        className="bg-white p-4 border border-neutral-200 rounded-xl shadow-md space-y-3"
                      >
                        {/* Header: Date & Marks */}
                        <div className="flex items-center justify-between border-b border-neutral-100 pb-3">
                          {/* Date */}
                          <div className="flex items-center text-sm font-medium text-dark">
                            <Calendar className="h-4 w-4 text-indigo-500 mr-2" />
                            <div>
                              <div className="font-semibold text-neutral-800">
                                {datePart}
                              </div>
                              <div className="text-xs text-neutral-500">
                                {timePart}
                              </div>
                            </div>
                          </div>

                          {/* Marks Scored / Total Marks */}
                          <div className="text-right">
                            <span
                              className={`text-xl font-extrabold ${getPerformanceColor(
                                percentage
                              )}`}
                            >
                              {marksScored}
                            </span>
                            <span className="text-sm text-neutral-500">
                              {" "}
                              / {totalMarks}
                            </span>
                            <div
                              className={`text-xs font-medium mt-0.5 ${getPerformanceColor(
                                percentage
                              )}`}
                            >
                              ({percentage}%)
                            </div>
                          </div>
                        </div>

                        {/* Body: Subject & Chapter */}
                        <div className="space-y-2">
                          {/* Subject */}
                          <div className="flex items-center text-sm text-neutral-700">
                            <BookOpen className="h-4 w-4 flex-shrink-0 mr-2 text-teal-600" />
                            <span className="font-semibold">Subject:</span>
                            <span className="ml-2 truncate">
                              {feedback.test_details?.subject || "N/A"}
                            </span>
                          </div>

                          {/* Chapter */}
                          <div className="flex items-center text-sm text-neutral-700">
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="h-4 w-4 flex-shrink-0 mr-2 text-teal-600 opacity-60"
                              fill="none"
                              viewBox="0 0 24 24"
                              stroke="currentColor"
                              strokeWidth={2}
                            >
                              <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-2.414-2.414A1 1 0 0015.586 6H7a2 2 0 00-2 2v11a2 2 0 002 2z"
                              />
                            </svg>
                            <span className="font-semibold">Chapter:</span>
                            <span
                              className="ml-2 truncate"
                              title={feedback.test_details?.chapter || ""}
                            >
                              {feedback.test_details?.chapter || "N/A"}
                            </span>
                          </div>
                        </div>

                        {/* Action Button */}
                        <div className="pt-2">
                          <Button
                            size="sm"
                            onClick={() =>
                              navigate(`/view-feedback/${feedback.id}`)
                            }
                            className="w-full justify-center flex items-center bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg py-2 transition duration-150"
                          >
                            <Eye className="h-4 w-4 mr-2" />
                            View Feedback
                          </Button>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* ---------------------------------------------------- */}
                {/* 2. DESKTOP VIEW (Table) - Hidden on screens < lg */}
                {/* ---------------------------------------------------- */}
                <div className="overflow-x-auto hidden lg:block">
                  <table className="min-w-full">
                    <thead className="bg-neutral-50 border-b border-neutral-200">
                      <tr>
                        <th className="px-6 py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                          Evaluation Date
                        </th>
                        <th className="px-6 py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                          Marks Scored
                        </th>
                        <th className="px-6 py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                          Total Marks
                        </th>
                        <th className="px-6 py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                          Subject
                        </th>
                        <th className="px-6 py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                          Topic
                        </th>
                        <th className="px-6 py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                          Difficulty Level
                        </th>
                        <th className="px-6 py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                          Performance
                        </th>
                        <th className="px-6 py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-neutral-200">
                      {filteredFeedbacks.map((feedback) => (
                        <tr
                          key={feedback.id}
                          className="hover:bg-neutral-50 transition-colors duration-150"
                        >
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <Calendar className="h-4 w-4 text-neutral-400 mr-2" />
                              <div>
                                <div className="text-sm font-medium text-dark">
                                  {
                                    formatDate(feedback.created_at).split(
                                      ","
                                    )[0]
                                  }
                                </div>
                                <div className="text-xs text-neutral-500">
                                  {
                                    formatDate(feedback.created_at).split(
                                      ","
                                    )[1]
                                  }
                                </div>
                              </div>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <Award className="h-4 w-4 text-primary-600 mr-2" />
                              <span
                                className={`text-lg font-bold ${getPerformanceColor(
                                  feedback.percentage || 0
                                )}`}
                              >
                                {feedback.marks_scored || 0}
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span className="text-sm font-medium text-dark">
                              {feedback.total_marks || 0}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex items-center">
                              <BookOpen className="h-4 w-4 text-secondary-600 mr-2" />
                              <span className="text-sm font-medium text-dark">
                                {feedback.test_details?.subject || "N/A"}
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4">
                            <div
                              className="text-sm text-dark max-w-xs truncate"
                              title={feedback.test_details?.chapter || ""}
                            >
                              {feedback.test_details?.chapter || "N/A"}
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <span
                              className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getDifficultyColor(
                                feedback.test_details?.difficulty_level || ""
                              )}`}
                            >
                              {feedback.test_details?.difficulty_level || "N/A"}
                            </span>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <div className="flex flex-col">
                              <span
                                className={`text-lg font-bold ${getPerformanceColor(
                                  feedback.percentage || 0
                                )}`}
                              >
                                {feedback.percentage || 0}%
                              </span>
                              <span
                                className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${getPerformanceBadgeColor(
                                  feedback.percentage || 0
                                )} mt-1`}
                              >
                                {getPerformanceLabel(feedback.percentage || 0)}
                              </span>
                            </div>
                          </td>
                          <td className="px-6 py-4 whitespace-nowrap">
                            <Button
                              size="sm"
                              onClick={() =>
                                navigate(`/view-feedback/${feedback.id}`)
                              }
                              className="flex items-center bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg py-2 transition duration-150"
                            >
                              <Eye className="h-4 w-4 mr-1" />
                              View Feedback
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}