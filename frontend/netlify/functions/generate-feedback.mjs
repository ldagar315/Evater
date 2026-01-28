export default async (request, context) => {
  const { corsHeadersFor } = await import('./_shared/cors.mjs')
  const cors = corsHeadersFor(request)

  // Handle CORS preflight requests
  if (request.method === 'OPTIONS') {
    return new Response(null, {
      status: 200,
      headers: {
        ...cors,
      },
    })
  }

  if (request.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: {
        'Content-Type': 'application/json',
        ...cors,
      },
    })
  }

  try {
    const body = await request.json()
    
    console.log('Netlify function received:', body)

    const authHeader = request.headers.get('authorization') || undefined
    
    // Forward the request to the external API
    const response = await fetch('https://ldagar315--evater-v1-wrapper.modal.run/api/gen_answer', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(authHeader ? { Authorization: authHeader } : {}),
      },
      body: JSON.stringify(body)
    })

    console.log('External API response status:', response.status)
    
    const data = await response.json()
    console.log('External API response data:', data)

    if (!response.ok) {
      return new Response(JSON.stringify(data), {
        status: response.status,
        headers: {
          'Content-Type': 'application/json',
          ...cors,
        },
      })
    }

    return new Response(JSON.stringify(data), {
      status: 200,
      headers: {
        'Content-Type': 'application/json',
        ...cors,
      },
    })
  } catch (error) {
    console.error('Error in generate-feedback function:', error)
    return new Response(JSON.stringify({ 
      error: 'Internal server error',
      details: error.message 
    }), {
      status: 500,
      headers: {
        'Content-Type': 'application/json',
        ...cors,
      },
    })
  }
}
