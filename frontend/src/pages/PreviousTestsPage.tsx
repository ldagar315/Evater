import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, FileText, Calendar, BookOpen, Eye, Search, Filter } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Select } from '../components/ui/Select'
import { Card, CardContent, CardHeader } from '../components/ui/Card'
import { Header } from '../components/layout/Header'
import { useAuthContext } from '../contexts/AuthContext'
import { supabase } from '../lib/supabase'
import { QuestionsCreated } from '../types'

export function PreviousTestsPage() {
  const navigate = useNavigate()
  const { user } = useAuthContext()
  
  const [tests, setTests] = useState<QuestionsCreated[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [searchTerm, setSearchTerm] = useState('')
  const [subjectFilter, setSubjectFilter] = useState('')
  const [gradeFilter, setGradeFilter] = useState('')

  useEffect(() => {
    if (user) {
      fetchTests()
    }
  }, [user])

  const fetchTests = async () => {
    try {
      setLoading(true)
      setError('')

      const { data, error } = await supabase
        .from('Questions_Created')
        .select('*')
        .eq('created_by', user!.id)
        .order('created_at', { ascending: false })

      if (error) throw error

      setTests(data || [])
    } catch (err) {
      console.error('Error fetching tests:', err)
      setError('Failed to load tests. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  // Get unique subjects and grades for filters
  const uniqueSubjects = [...new Set(tests.map(test => test.subject).filter(Boolean))].sort()
  const uniqueGrades = [...new Set(tests.map(test => test.grade).filter(Boolean))].sort((a, b) => {
    const numA = parseInt(a!)
    const numB = parseInt(b!)
    return numA - numB
  })

  // Filter tests based on search and filters
  const filteredTests = tests.filter(test => {
    const matchesSearch = searchTerm === '' || 
      (test.subject && test.subject.toLowerCase().includes(searchTerm.toLowerCase())) ||
      (test.chapter && test.chapter.toLowerCase().includes(searchTerm.toLowerCase()))
    
    const matchesSubject = subjectFilter === '' || test.subject === subjectFilter
    const matchesGrade = gradeFilter === '' || test.grade === gradeFilter

    return matchesSearch && matchesSubject && matchesGrade
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

  const getDifficultyColor = (difficulty: string | null | undefined) => {
    if (!difficulty) return 'bg-neutral-100 text-neutral-800'
    
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

  const getQuestionCount = (test: QuestionsCreated): number => {
    if (!test.test) return 0
    if (Array.isArray(test.test)) return test.test.length
    if (typeof test.test === 'object' && test.test.questions) {
      return Array.isArray(test.test.questions) ? test.test.questions.length : 0
    }
    return 0
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />
        <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-center py-12">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
              <p className="text-neutral-600">Loading your tests...</p>
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
            <div className = "mb-6">
            <div className="flex items-center mb-2">

              <h1 className="text-xl font-semibold text-dark sm:text-2xl">
                Previous Tests
              </h1>
            </div>
            <p className="text-sm text-neutral-600">
              View and manage all your previously created tests
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
        {/* Summary Stats */}
        {filteredTests.length > 0 && (
          <Card className="mt-6 mb-6">
            <CardContent className="p-6">
              <h3 className="text-lg font-semibold text-dark mb-4">
                Quick Stats
              </h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-primary-50 rounded-lg">
                  <div className="text-xl font-bold text-primary-600">
                    {filteredTests.length}
                  </div>
                  <div className="text-sm text-primary-800">Total Tests</div>
                </div>
                <div className="text-center p-4 bg-primary-50 rounded-lg">
                  <div className="text-xl font-bold text-primary-600">
                    {uniqueSubjects.length}
                  </div>
                  <div className="text-sm text-primary-800">Subjects</div>
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
              <h3 className="text-lg font-semibold text-dark">Filter Tests</h3>
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
                value={gradeFilter}
                onChange={(e) => setGradeFilter(e.target.value)}
                options={[
                  { value: "", label: "All Grades" },
                  ...uniqueGrades.map((grade) => ({
                    value: grade,
                    label: `Grade ${grade}`,
                  })),
                ]}
              />
              <Button
                variant="outline"
                onClick={() => {
                  setSearchTerm("");
                  setSubjectFilter("");
                  setGradeFilter("");
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
            Showing {filteredTests.length} of {tests.length} tests
          </p>
          <div className="flex items-center space-x-2 text-sm text-neutral-600">
            <Calendar className="h-4 w-4" />
            <span>Sorted by creation date (newest first)</span>
          </div>
        </div>
        {/* Tests Table */}
        <Card className = "px-2 py-2 lg:px-6 lg:py-4 ">
          <CardContent className="p-0">
            {/* --- EMPTY STATE (Renders regardless of screen size) --- */}
            {filteredTests.length === 0 ? (
              <div className="p-12 text-center">
                <FileText className="h-12 w-12 text-neutral-400 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-dark mb-2">
                  {tests.length === 0
                    ? "No Tests Created Yet"
                    : "No Tests Found"}
                </h3>
                <p className="text-neutral-600 mb-4">
                  {tests.length === 0
                    ? "You haven't created any tests yet. Create your first test to get started!"
                    : "No tests match your current filters. Try adjusting your search criteria."}
                </p>
                <Button
                  onClick={() => navigate("/create-test")}
                  className="flex items-center"
                >
                  <BookOpen className="h-4 w-4 mr-2" />
                  {tests.length === 0
                    ? "Create Your First Test"
                    : "Create New Test"}
                </Button>
              </div>
            ) : (
              <>
                {/* ============================================== */}
                {/* 1. TABLE LAYOUT (For Large Screens/Laptops) */}
                {/* Hidden by default, shown from 'lg' breakpoint up */}
                {/* ============================================== */}
                <div className="hidden lg:block">
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="bg-neutral-50 border-b border-neutral-200">
                        <tr>
                          <th className="px-2 py-2 lg:px-6 lg:py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                            Created Date
                          </th>
                          <th className="px-2 py-2 lg:px-6 lg:py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                            Subject
                          </th>
                          <th className="px-2 py-2 lg:px-6 lg:py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                            Topic/Chapter
                          </th>
                          <th className="px-2 py-2 lg:px-6 lg:py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                            Grade
                          </th>
                          <th className="px-2 py-2 lg:px-6 lg:py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                            Difficulty
                          </th>
                          <th className="px-2 py-2 lg:px-6 lg:py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                            Questions
                          </th>
                          <th className="px-2 py-2 lg:px-6 lg:py-4 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                            Actions
                          </th>
                        </tr>
                      </thead>
                      <tbody className="bg-white divide-y divide-neutral-200">
                        {filteredTests.map((test) => (
                          <tr
                            key={test.id}
                            className="hover:bg-neutral-50 transition-colors duration-150"
                          >
                            <td className="px-2 py-2 lg:px-6 lg:py-4 whitespace-nowrap">
                              <div className="flex items-center">
                                <Calendar className="h-4 w-4 text-neutral-400 mr-2" />
                                <div>
                                  <div className="text-sm font-medium text-dark">
                                    {formatDate(test.created_at).split(",")[0]}
                                  </div>
                                  <div className="text-xs text-neutral-500">
                                    {formatDate(test.created_at).split(",")[1]}
                                  </div>
                                </div>
                              </div>
                            </td>
                            <td className="px-2 py-2 lg:px-6 lg:py-4 whitespace-nowrap">
                              <div className="flex items-center">
                                <BookOpen className="h-4 w-4 text-primary-600 mr-2" />
                                <span className="text-sm font-medium text-dark">
                                  {test.subject || "N/A"}
                                </span>
                              </div>
                            </td>
                            <td className="px-2 py-2 lg:px-6 lg:py-4">
                              <div
                                className="text-sm text-dark max-w-xs truncate"
                                title={test.chapter || ""}
                              >
                                {test.chapter || "N/A"}
                              </div>
                            </td>
                            <td className="px-2 py-2 lg:px-6 lg:py-4 whitespace-nowrap">
                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                                Grade {test.grade || "N/A"}
                              </span>
                            </td>
                            <td className="px-2 py-2 lg:px-6 lg:py-4 whitespace-nowrap">
                              <span
                                className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getDifficultyColor(
                                  test.difficulty_level
                                )}`}
                              >
                                {test.difficulty_level || "N/A"}
                              </span>
                            </td>
                            <td className="px-2 py-2 lg:px-6 lg:py-4 whitespace-nowrap">
                              <div className="text-sm text-dark font-medium">
                                {getQuestionCount(test)} questions
                              </div>
                            </td>
                            <td className="px-2 py-2 lg:px-6 lg:py-4 whitespace-nowrap">
                              <Button
                                size="sm"
                                onClick={() =>
                                  navigate(`/view-test/${test.id}`)
                                }
                                className="flex items-center"
                              >
                                <Eye className="h-4 w-4 mr-1" />
                                View Test
                              </Button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* ============================================== */}
                {/* 2. CARD LAYOUT (For Mobile/Small Screens) */}
                {/* Shown by default, hidden from 'lg' breakpoint up */}
                {/* ============================================== */}
                <div className="lg:hidden p-1 space-y-4">
                  {filteredTests.map((test) => (
                    <div
                      key={test.id}
                      className="border border-neutral-200 rounded-lg p-4 shadow-sm"
                    >
                      {/* Row 1: Subject, Grade, Chapter */}
                      <div className="flex justify-between items-start mb-2">
                        <div className="flex-1 min-w-0 pr-2">
                          <p className="text-xs font-medium text-neutral-500 uppercase">
                            {test.subject || ""}
                          </p>
                          <div
                            className="text-xs font-semibold text-dark truncate"
                          >
                            {test.chapter || "N/A"}
                          </div>
                        </div>
                        <span className="flex-shrink-0 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-200 text-primary-800">
                          Grade {test.grade || "N/A"}
                        </span>
                      </div>

                      {/* Row 2: Date, Button */}
                      <div className="flex justify-between items-center pt-2 border-t border-neutral-100">
                        <div className="flex items-center text-sm text-neutral-600">
                          <Calendar className="h-4 w-4 mr-1 text-neutral-400" />
                          <span className="text-xs">
                            {formatDate(test.created_at).split(",")[0]}
                          </span>
                        </div>
                        <Button
                          size="sm"
                          onClick={() => navigate(`/view-test/${test.id}`)}
                          className="flex items-center flex-shrink-0"
                        >
                          <Eye className="h-4 w-4 mr-1" />
                          View Test
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}