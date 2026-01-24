export interface BlogPost {
  id: string
  title: string
  slug: string
  excerpt: string
  content: string
  featured_image: string
  author: {
    name: string
    avatar?: string
    bio?: string
  }
  category: string
  tags: string[]
  published_date: string
  read_time: number
  is_featured: boolean
  status: 'draft' | 'published' | 'archived'
  seo: {
    meta_title?: string
    meta_description?: string
  }
}

export interface BlogCategory {
  id: string
  name: string
  slug: string
  description: string
  post_count: number
}