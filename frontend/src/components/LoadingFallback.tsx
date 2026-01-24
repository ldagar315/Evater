import React from 'react'

export function LoadingFallback() {
  return (
    <div className="min-h-screen bg-cream flex items-center justify-center">
      <div className="text-center">
        <div className="inline-flex items-center justify-center w-24 h-16 mb-6 animate-pulse">
          <img 
            src="/Evater_logo_2.png" 
            alt="Evater Logo" 
            className="h-16 w-auto object-contain"
          />
        </div>
        <h2 className="text-2xl font-bold text-dark mb-2">Loading Evater</h2>
        <p className="text-neutral-600 mb-4">Please wait while we prepare your learning experience...</p>
        <div className="flex justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        </div>
      </div>
    </div>
  )
}