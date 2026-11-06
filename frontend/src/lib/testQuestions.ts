// Client version sent to the backend. The answer-key integrity backend
// (key-free responses) rejects older clients with a visible 409 rather than
// serving them data they would silently mis-grade.
export const CLIENT_VERSION = 'web/2'

// Stored Questions_Created.test shapes:
//   - bare array of questions (legacy rows, may still carry is_correct)
//   - { questions: [...] } (older object format)
//   - { v: 2, questions: [...] } (written after the key-integrity change)
// Old clients guard every read with Array.isArray(test), so an object shape
// is fail-VISIBLE for them (error/empty state) - never the silent
// first-option-correct fallback. New clients must accept all three.
export function unwrapTestQuestions(test: any): any[] {
  if (Array.isArray(test)) return test
  if (test && typeof test === 'object' && Array.isArray(test.questions)) return test.questions
  return []
}
