import React from 'react'
import { useNavigate } from 'react-router-dom'
import { BookOpen, ArrowRight } from 'lucide-react'
import { Button } from '../ui/Button'
import { Card, CardContent } from '../ui/Card'
import { BlogCard } from './BlogCard'
import { getRecentPosts } from '../../data/blogPosts'

interface BlogSectionProps {
  showInDashboard?: boolean
}

export function BlogSection({ showInDashboard = false }: BlogSectionProps) {
  const navigate = useNavigate()
  const recentPosts = getRecentPosts(showInDashboard ? 2 : 3)

  const handlePostClick = (slug: string) => {
    console.log('BlogSection handlePostClick called with slug:', slug)
    navigate(`/blog/${slug}`)
  }

  if (showInDashboard) {
    return (
      <Card className="bg-gradient-to-br from-purple-50 to-pink-50 border-purple-200">
        <CardContent className="p-6">
          <div className="flex items-center mb-4">
            <BookOpen className="h-6 w-6 text-purple-600 mr-2" />
            <h3 className="text-lg font-semibold text-dark">Educational Insights</h3>
          </div>
          
          <div className="space-y-4 mb-4">
            {recentPosts.map((post) => (
              <div 
                key={post.id}
                className="cursor-pointer group"
                onClick={() => handlePostClick(post.slug)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    handlePostClick(post.slug)
                  }
                }}
              >
                <h4 className="font-medium text-dark group-hover:text-purple-600 transition-colors duration-200 line-clamp-2 mb-1">
                  {post.title}
                </h4>
                <p className="text-sm text-neutral-600 line-clamp-2 mb-2">
                  {post.excerpt}
                </p>
                <div className="flex items-center text-xs text-neutral-500">
                  <span>{post.category}</span>
                  <span className="mx-2">•</span>
                  <span>{post.read_time} min read</span>
                </div>
              </div>
            ))}
          </div>
          
          <Button
            onClick={() => navigate('/blog')}
            variant="outline"
            size="sm"
            className="w-full flex items-center justify-center text-purple-600 border-purple-300 hover:bg-purple-50"
          >
            View All Blog Posts
            <ArrowRight className="h-4 w-4 ml-1" />
          </Button>
        </CardContent>
      </Card>
    )
  }

  return (
    <section className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl lg:text-5xl font-bold text-dark mb-6">
            Latest from Our Blog
          </h2>
          <p className="text-xl text-neutral-600 max-w-3xl mx-auto">
            Insights, tips, and updates from the Evater team to help you make the most of your educational journey
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-12">
          {recentPosts.map((post) => (
            <BlogCard key={post.id} post={post} featured />
          ))}
        </div>
        
        <div className="text-center">
          <Button
            onClick={() => navigate('/blog')}
            size="lg"
            className="px-8 py-3 text-lg font-medium shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200"
          >
            View All Posts
            <ArrowRight className="h-5 w-5 ml-2" />
          </Button>
        </div>
      </div>
    </section>
  )
}