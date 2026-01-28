import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Upload, FileText, ArrowLeft, CheckCircle } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card, CardContent, CardHeader } from '../components/ui/Card'
import { Header } from '../components/layout/Header'
import { useAuthContext } from '../contexts/AuthContext'
import { useAppState } from '../contexts/AppStateContext'
import { supabase } from '../lib/supabase'
import { QuestionsCreated, FeedbackResponse } from '../types'

interface ApiFeedbackResponse {
  merged: FeedbackResponse[]
}

export function SubmitFeedbackPage() {
  const { testId } = useParams()
  const navigate = useNavigate()
  const { user } = useAuthContext()
  const { appState, setLastGeneratedFeedback } = useAppState()
  
  const [test, setTest] = useState<QuestionsCreated | null>(null)
  const [files, setFiles] = useState<File[]>([])
  const [uploading, setUploading] = useState(false)
  const [evaluating, setEvaluating] = useState(false)
  const [error, setError] = useState('')
  const [uploadedFileUrls, setUploadedFileUrls] = useState<string[]>([])
  const [uploadCompleted, setUploadCompleted] = useState(false)
  const [uploadingToBucket, setUploadingToBucket] = useState(false)

  useEffect(() => {
    if (testId) {
      fetchTest()
    } else if (appState.last_generated_test) {
      setTest(appState.last_generated_test)
    } else {
      fetchLatestTest()
    }
  }, [testId, appState.last_generated_test])

  const fetchTest = async () => {
    try {
      const { data, error } = await supabase
        .from('Questions_Created')
        .select('*')
        .eq('id', parseInt(testId!))
        .single()

      if (error) throw error
      setTest(data)
    } catch (err) {
      setError('Failed to load test')
    }
  }

  const fetchLatestTest = async () => {
    try {
      const { data, error } = await supabase
        .from('Questions_Created')
        .select('*')
        .order('created_at', { ascending: false })
        .limit(1)
        .single()

      if (error) throw error
      setTest(data)
    } catch (err) {
      setError('No tests found')
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || [])
    setFiles(selectedFiles)
  }

  const uploadToBucket = async () => {
    if (files.length === 0) {
      setError('Please select files first')
      return
    }

    setUploadingToBucket(true)
    setError('')
    
    try {
      const bucketName = 'question-papers-test'
      const folderName = 'question-test-v1'
      const uploadedUrls: string[] = []

      // Upload each file to the specified bucket and folder
      for (let i = 0; i < files.length; i++) {
        const file = files[i]
        const timestamp = Date.now()
        const fileName = `${folderName}/${timestamp}_${i + 1}_${file.name}`

        const { data, error } = await supabase.storage
          .from(bucketName)
          .upload(fileName, file, {
            cacheControl: '3600',
            upsert: false
          })

        if (error) throw error

        // Get the public URL for the uploaded file
        const { data: urlData } = supabase.storage
          .from(bucketName)
          .getPublicUrl(fileName)

        if (urlData?.publicUrl) {
          uploadedUrls.push(urlData.publicUrl)
        }
      }

      setUploadedFileUrls(uploadedUrls)
      setUploadCompleted(true)
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to upload files to bucket')
    } finally {
      setUploadingToBucket(false)
    }
  }

  // Helper function to capitalize difficulty
  const capitalizeDifficulty = (difficulty: string): 'Easy' | 'Medium' | 'Hard' => {
    const normalized = difficulty.toLowerCase()
    switch (normalized) {
      case 'easy':
        return 'Easy'
      case 'medium':
        return 'Medium'
      case 'hard':
        return 'Hard'
      default:
        return 'Medium' // fallback
    }
  }

  const callGenerateFeedbackAPI = async (imageUrls: string[], questions: any[]): Promise<FeedbackResponse[]> => {
    // The backend expects: InputDataAnswer with image_url: List[str] and questions: Dict[str, Any]
    // Based on the error, the backend is doing: questions['questions'] and then iterating over that
    // So we need to pass the questions array directly under the 'questions' key
    const payload = {
      image_url: imageUrls,  // List[str]
      questions: {           // Dict[str, Any]
        questions: questions // The actual questions array goes here
      }
    }

    console.log('API Payload:', JSON.stringify(payload, null, 2))

    // Use Netlify function in production, proxy in development
    const apiUrl = process.env.NODE_ENV === 'production' 
      ? '/.netlify/functions/generate-feedback'
      : '/api/external/api/gen_answer'

    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload)
    })

    if (!response.ok) {
      if (response.status === 400) {
        throw new Error('Bad Request: Missing image_url or questions in the request')
      } else if (response.status === 422) {
        throw new Error('Unprocessable Entity: Invalid data format')
      } else if (response.status === 500) {
        throw new Error('Internal Server Error: Failed in OCR or feedback generation')
      } else {
        throw new Error(`API error: ${response.status} - ${response.statusText}`)
      }
    }

    const data: ApiFeedbackResponse = await response.json()
    return data.merged
  }

  const handleEvaluate = async () => {
    if (!test) {
      setError('Test data not available')
      return
    }

    // Check if we have either uploaded files or bucket URLs
    if (files.length === 0 && uploadedFileUrls.length === 0) {
      setError('Please select files or upload to bucket first')
      return
    }

    setEvaluating(true)
    setError('')

    try {
      let imageUrls: string[] = []

      // If files were uploaded to bucket, use those URLs
      if (uploadedFileUrls.length > 0) {
        imageUrls = uploadedFileUrls
      } else {
        // Upload files to answer-sheets bucket and get URLs
        const uploadPromises = files.map(async (file, index) => {
          const fileName = `${user!.id}/${test.id}/answer_sheet_${index + 1}_${Date.now()}.${file.name.split('.').pop()}`
          
          const { data, error } = await supabase.storage
            .from('answer-sheets')
            .upload(fileName, file)

          if (error) throw error

          // Get public URL
          const { data: urlData } = supabase.storage
            .from('answer-sheets')
            .getPublicUrl(fileName)

          return urlData.publicUrl
        })

        imageUrls = await Promise.all(uploadPromises)
      }

      // Prepare questions as an array (not dictionary)
      const questionsArray = Array.isArray(test.test) ? test.test : []
      
      // Ensure each question has the required fields for the backend Question model
      // IMPORTANT: Capitalize the difficulty field to match backend expectations
      const formattedQuestions = questionsArray.map((question: any) => ({
        question_text: question.question_text,
        question_type: question.question_type,
        question_number: question.question_number,
        maximum_marks: question.maximum_marks,
        difficulty: capitalizeDifficulty(question.difficulty), // Capitalize this field
        contains_math_expression: question.contains_math_expression,
        options: question.options || null
      }))

      console.log('Calling API with:', {
        imageUrls: imageUrls.length,
        questionsCount: formattedQuestions.length,
        sampleQuestion: formattedQuestions[0]
      })

      // Call the Generate Feedback API with the questions array
      const feedbackResponse = await callGenerateFeedbackAPI(imageUrls, formattedQuestions)

      // Save answer sheet images to database
      const { data: answerSheetData, error: answerSheetError } = await supabase
        .from('AnswerSheetImages')
        .insert([{
          uploaded_by: user!.id,
          for_test: test.id,
          image_url: imageUrls
        }])
        .select()
        .single()

      if (answerSheetError) throw answerSheetError

      // Save feedback to database
      const { data: feedbackData, error: feedbackError } = await supabase
        .from('FeedbackTest')
        .insert([{
          given_by: user!.id,
          for_test: test.id,
          feedback: feedbackResponse
        }])
        .select()
        .single()

      if (feedbackError) throw feedbackError

      // Store in app state
      setLastGeneratedFeedback(feedbackData)

      navigate(`/view-feedback/${feedbackData.id}`)
    } catch (err) {
      console.error('Evaluation error:', err)
      setError(err instanceof Error ? err.message : 'Failed to evaluate answer sheets. Please try again.')
    } finally {
      setEvaluating(false)
    }
  }

  if (!test) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-cream">
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

        <Card className="mb-6">
          <CardHeader>
            <h2 className="text-2xl font-bold text-dark">Submit Answer Sheets</h2>
            <p className="text-sm text-neutral-600">
              Upload images of completed answer sheets for: <strong>{test.subject} - {test.chapter}</strong>
            </p>
          </CardHeader>
        </Card>

        <Card className="mb-6">
          <CardContent className="p-6">
            <div className="border-2 border-dashed border-neutral-300 rounded-lg p-8 text-center">
              <Upload className="h-12 w-12 text-neutral-400 mx-auto mb-4" />
              <div className="space-y-2">
                <p className="text-lg font-medium text-dark">
                  Upload Answer Sheet Images
                </p>
                <p className="text-sm text-neutral-600">
                  Select multiple images (JPG, PNG, PDF) of the completed answer sheets
                </p>
              </div>
              
              <div className="flex flex-col sm:flex-row gap-4 mt-6 justify-center">
                <div>
                  <input
                    type="file"
                    multiple
                    accept="image/*,.pdf"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="file-upload"
                  />
                  <label
                    htmlFor="file-upload"
                    className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-primary-600 hover:bg-primary-700 cursor-pointer"
                  >
                    Choose Files
                  </label>
                </div>
                
                <Button
                  onClick={uploadToBucket}
                  loading={uploadingToBucket}
                  variant="secondary"
                  className="inline-flex items-center"
                  disabled={files.length === 0}
                >
                  {uploadingToBucket ? 'Uploading to Bucket...' : 'Upload to Bucket'}
                </Button>
              </div>
            </div>

            {uploadCompleted && (
              <div className="mt-6 p-4 bg-primary-50 border border-primary-200 rounded-lg">
                <div className="flex items-center mb-2">
                  <CheckCircle className="h-5 w-5 text-primary-600 mr-2" />
                  <h3 className="text-sm font-medium text-primary-800">Upload Completed!</h3>
                </div>
                <p className="text-sm text-primary-700 mb-2">
                  Successfully uploaded {uploadedFileUrls.length} files to bucket: question-papers-test/question-test-v1
                </p>
                <p className="text-xs text-primary-600">
                  Files stored in variable: uploadedFileUrls
                </p>
              </div>
            )}

            {uploadedFileUrls.length > 0 && (
              <div className="mt-6">
                <h3 className="text-sm font-medium text-dark mb-3">
                  Uploaded Files ({uploadedFileUrls.length})
                </h3>
                <div className="max-h-40 overflow-y-auto space-y-2">
                  {uploadedFileUrls.map((url, index) => {
                    const fileName = url.split('/').pop() || `File ${index + 1}`
                    return (
                      <div key={index} className="flex items-center p-3 bg-neutral-50 rounded-lg">
                        <FileText className="h-5 w-5 text-neutral-400 mr-3" />
                        <div className="flex-1">
                          <p className="text-sm font-medium text-dark">{fileName}</p>
                          <p className="text-xs text-neutral-500 truncate">{url}</p>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {files.length > 0 && !uploadCompleted && (
              <div className="mt-6">
                <h3 className="text-sm font-medium text-dark mb-3">
                  Selected Files ({files.length})
                </h3>
                <div className="space-y-2">
                  {files.map((file, index) => (
                    <div key={index} className="flex items-center p-3 bg-neutral-50 rounded-lg">
                      <FileText className="h-5 w-5 text-neutral-400 mr-3" />
                      <div className="flex-1">
                        <p className="text-sm font-medium text-dark">{file.name}</p>
                        <p className="text-sm text-neutral-500">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {error && (
          <Card className="mb-6">
            <CardContent className="p-4">
              <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardContent className="p-6 text-center">
            <h3 className="text-lg font-semibold text-dark mb-2">
              Ready to Evaluate?
            </h3>
            <p className="text-neutral-600 mb-4">
              Our AI will analyze the answer sheets and provide detailed feedback using OCR and intelligent evaluation
            </p>
            <Button
              onClick={handleEvaluate}
              disabled={files.length === 0 && uploadedFileUrls.length === 0}
              loading={evaluating}
              size="lg"
            >
              {evaluating ? 'Evaluating Answer Sheets...' : 'Evaluate Answer Sheets'}
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}