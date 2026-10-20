import { Link } from 'react-router-dom'
import { Seo } from '../components/seo/Seo'

export function NotFoundPage() {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center px-4 text-center">
      <Seo
        title="Page not found | Evater"
        description="The page you are looking for does not exist."
        path="/404"
        robots="noindex, nofollow"
      />
      <p className="text-6xl font-bold text-gray-300 mb-4">404</p>
      <h1 className="text-2xl font-bold text-gray-900 mb-2">Page not found</h1>
      <p className="text-gray-600 mb-6">
        The link may be broken or the page may have been moved.
      </p>
      <Link to="/" className="text-blue-600 hover:underline">
        Go to the Evater home page
      </Link>
    </div>
  )
}
