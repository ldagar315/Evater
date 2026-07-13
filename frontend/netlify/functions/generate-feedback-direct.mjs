export default async (request, context) => {
  const { corsDecision } = await import('./_shared/cors.mjs')
  const { modalApiUrl } = await import('./_shared/modal.mjs')
  const { origin, allowed, headers: cors } = corsDecision(request)

  if (origin && !allowed) {
    return new Response(JSON.stringify({ error: 'CORS origin not allowed' }), {
      status: 403,
      headers: { 'Content-Type': 'application/json' },
    })
  }

  if (request.method === 'OPTIONS') {
    return new Response(null, { status: 200, headers: cors })
  }

  if (request.method !== 'POST') {
    return new Response(JSON.stringify({ error: 'Method not allowed' }), {
      status: 405,
      headers: { 'Content-Type': 'application/json', ...cors },
    })
  }

  try {
    const authHeader = request.headers.get('authorization') || undefined
    const response = await fetch(`${modalApiUrl}/api/gen_feedback_direct`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(authHeader ? { Authorization: authHeader } : {}),
      },
      body: JSON.stringify(await request.json()),
    })

    const data = await response.json()
    return new Response(JSON.stringify(data), {
      status: response.status,
      headers: { 'Content-Type': 'application/json', ...cors },
    })
  } catch (error) {
    return new Response(JSON.stringify({
      error: 'Internal server error',
      details: error instanceof Error ? error.message : 'Unknown error',
    }), {
      status: 500,
      headers: { 'Content-Type': 'application/json', ...cors },
    })
  }
}
