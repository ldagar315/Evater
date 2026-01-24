import { useEffect, useState } from 'react'
import { supabase } from '../lib/supabase'
import { UserProfile } from '../types'
import { BYPASS_AUTH, buildDevProfile, logAuthBypassWarning } from '../lib/auth/devBypass'

export function useProfile(userId: string | undefined) {
  const [profile, setProfile] = useState<UserProfile | null>(
    BYPASS_AUTH ? buildDevProfile({ created_by: userId }) : null
  )
  const [loading, setLoading] = useState(!BYPASS_AUTH)

  useEffect(() => {
    if (BYPASS_AUTH) {
      logAuthBypassWarning('useProfile bootstrap')
      setProfile(buildDevProfile({ created_by: userId }))
      setLoading(false)
      return
    }

    if (!userId) {
      setLoading(false)
      return
    }

    fetchProfile(userId)
  }, [userId])

  const fetchProfile = async (targetUserId: string) => {
    if (!targetUserId) return

    try {
      const { data, error } = await supabase
        .from('Users')
        .select('*')
        .eq('created_by', targetUserId)
        .single()

      if (error && error.code !== 'PGRST116') throw error
      setProfile(data)
    } catch (error) {
      console.error('Error fetching profile:', error)
    } finally {
      setLoading(false)
    }
  }

  const createProfile = async (profileData: Omit<UserProfile, 'id' | 'created_at' | 'created_by'>) => {
    if (BYPASS_AUTH) {
      const mockProfile = buildDevProfile({
        ...profileData,
        name: profileData.name ?? profileData.user_name ?? 'Dev User',
        user_name: profileData.user_name ?? profileData.name ?? 'Dev User',
        created_by: userId,
      })
      setProfile(mockProfile)
      logAuthBypassWarning('createProfile noop')
      return { data: mockProfile, error: null }
    }

    if (!userId) return { error: 'No user ID' }

    try {
      // Get the current user's email from auth
      const { data: { user } } = await supabase.auth.getUser()
      
      const profilePayload = {
        user_name: profileData.name || profileData.user_name, // Use name as user_name
        email: user?.email || profileData.email, // Use auth email if available
        grade: profileData.grade, // Only use grade, remove class_level
        school: profileData.school,
        created_by: userId, // Set to the authenticated user's ID
        credits: 100 // Set default credits for new users
      }

      console.log('Creating profile with payload:', profilePayload)

      const { data, error } = await supabase
        .from('Users')
        .insert([profilePayload])
        .select()
        .single()

      if (error) {
        console.error('Profile creation error:', error)
        throw error
      }
      
      setProfile(data)
      return { data, error: null }
    } catch (error) {
      console.error('Profile creation failed:', error)
      return { data: null, error }
    }
  }

  const updateProfile = async (updates: Partial<UserProfile>) => {
    if (BYPASS_AUTH) {
      const mockProfile = buildDevProfile({
        ...profile,
        ...updates,
        created_by: userId,
      })
      setProfile(mockProfile)
      logAuthBypassWarning('updateProfile noop')
      return { data: mockProfile, error: null }
    }

    if (!userId || !profile) return { error: 'No user ID or profile' }

    try {
      // Remove class_level from updates if it exists
      const { class_level, ...cleanUpdates } = updates as any
      
      const { data, error } = await supabase
        .from('Users')
        .update(cleanUpdates)
        .eq('id', profile.id)
        .select()
        .single()

      if (error) throw error
      setProfile(data)
      return { data, error: null }
    } catch (error) {
      return { data: null, error }
    }
  }

  return {
    profile,
    loading,
    createProfile,
    updateProfile,
  }
}
