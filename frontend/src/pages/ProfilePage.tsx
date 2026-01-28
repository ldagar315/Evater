import React, { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { Select } from '../components/ui/Select'
import { Card, CardContent, CardHeader } from '../components/ui/Card'
import { Header } from '../components/layout/Header'
import { useAuthContext } from '../contexts/AuthContext'
import { useProfile } from '../hooks/useProfile'

export function ProfilePage() {
  const navigate = useNavigate()
  const { user } = useAuthContext()
  const { profile, loading, createProfile, updateProfile } = useProfile(user?.id)
  
  const [name, setName] = useState('')
  const [selectedGrade, setSelectedGrade] = useState('')
  const [school, setSchool] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')

  const gradeOptions = Array.from({ length: 12 }, (_, i) => ({
    value: i + 1,
    label: `Grade ${i + 1}`
  }))

  useEffect(() => {
    if (profile) {
      setName(profile.name || profile.user_name || '')
      setSelectedGrade(profile.grade?.toString() || '')
      setSchool(profile.school || '')
    }
  }, [profile])

  useEffect(() => {
    if (!loading && profile) {
      navigate('/home')
    }
  }, [loading, profile, navigate])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setSubmitting(true)

    try {
      // Validate required fields
      if (!name.trim() || !selectedGrade || !school.trim()) {
        throw new Error('Please fill in all required fields')
      }

      const profileData = {
        name: name.trim(),
        user_name: name.trim(), // Set both name and user_name
        grade: parseInt(selectedGrade),
        school: school.trim(),
        email: user?.email || '',
      }

      console.log('Submitting profile data:', profileData)

      const { error } = profile
        ? await updateProfile(profileData)
        : await createProfile(profileData)

      if (error) {
        console.error('Profile save error:', error)
        setError(error.message || 'Failed to save profile. Please try again.')
      } else {
        console.log('Profile saved successfully')
        navigate('/home')
      }
    } catch (err) {
      console.error('Profile submission error:', err)
      setError(err instanceof Error ? err.message : 'An unexpected error occurred')
    } finally {
      setSubmitting(false)
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
            <p className="text-neutral-600">Loading profile...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-cream">
      <Header />
      
      <div className="max-w-2xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <Card>
          <CardHeader>
            <h2 className="text-2xl font-bold text-dark">
              {profile ? 'Update Profile' : 'Complete Your Profile'}
            </h2>
            <p className="text-sm text-neutral-600">
              {profile 
                ? 'Update your information below' 
                : 'Please provide your details to get started with Evater'
              }
            </p>
            {user?.email && (
              <p className="text-sm text-neutral-500">
                Signed in as: {user.email}
              </p>
            )}
          </CardHeader>
          
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <Input
                label="Full Name *"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                placeholder="Enter your full name"
              />
              
              <Select
                label="Grade *"
                value={selectedGrade}
                onChange={(e) => setSelectedGrade(e.target.value)}
                options={[{ value: '', label: 'Select your grade' }, ...gradeOptions]}
                required
              />
              
              <Input
                label="School *"
                value={school}
                onChange={(e) => setSchool(e.target.value)}
                required
                placeholder="Enter your school name"
              />
              
              {error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                  <div className="flex items-start">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <h3 className="text-sm font-medium text-red-800">Profile Creation Error</h3>
                      <p className="text-sm text-red-700 mt-1">{error}</p>
                    </div>
                  </div>
                </div>
              )}
              
              <div className="flex space-x-4">
                {profile && (
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => navigate('/home')}
                    className="flex-1"
                  >
                    Cancel
                  </Button>
                )}
                <Button
                  type="submit"
                  className={profile ? "flex-1" : "w-full"}
                  loading={submitting}
                  disabled={!name.trim() || !selectedGrade || !school.trim()}
                >
                  {submitting 
                    ? (profile ? 'Updating...' : 'Creating Profile...') 
                    : (profile ? 'Update Profile' : 'Create Profile')
                  }
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Debug info in development */}
        {process.env.NODE_ENV === 'development' && (
          <Card className="mt-6">
            <CardContent className="p-4">
              <h3 className="text-sm font-medium text-neutral-700 mb-2">Debug Info</h3>
              <div className="text-xs text-neutral-600 space-y-1">
                <p>User ID: {user?.id || 'Not available'}</p>
                <p>User Email: {user?.email || 'Not available'}</p>
                <p>Profile exists: {profile ? 'Yes' : 'No'}</p>
                <p>Loading: {loading ? 'Yes' : 'No'}</p>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}