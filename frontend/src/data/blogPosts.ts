import { BlogPost, BlogCategory } from '../types/blog'

export const blogCategories: BlogCategory[] = [
  {
    id: '1',
    name: 'Study Tips',
    slug: 'study-tips',
    description: 'Effective strategies for better learning outcomes',
    post_count: 1
  }
]

export const blogPosts: BlogPost[] = [
  {
    id: '1',
    title: 'How Students in Middle School Should Learn',
    slug: 'how-students-in-middle-school-should-learn',
    excerpt: 'Are you a middle school student in India, struggling to keep up with your studies in classes 6-10? You\'re not alone! This comprehensive guide provides effective learning strategies tailored for Indian middle school students.',
    content: `# How Students in Middle School Should Learn

Are you a middle school student in India, struggling to keep up with your studies in classes 6-10? You're not alone! Many students face challenges like exam pressure, tricky subjects, and managing time. The good news? With the right learning strategies, you can make studying easier, more fun, and set yourself up for success.

This blog is your guide to learning effectively, tailored for middle school students in India. Let's dive into tips and tricks that will help you shine in your studies while keeping things simple and stress-free.

## Understanding the Indian Education System

Middle school in India, covering classes 6 to 10, is an exciting yet challenging phase. You're learning subjects like mathematics, science, social studies, and languages, all while preparing for the big milestone: the class 10 board exams. These exams are super important because they can shape your future academic path.

But here's the catch—there's a lot of competition, and the pressure can feel overwhelming. Don't worry! By learning smart, you can tackle these challenges and come out on top.

## Effective Learning Strategies

### Time Management

Time is your best friend if you use it wisely. Here's how:

**Make a Study Schedule**: Plan your week ahead. Set aside time for each subject, homework, and even play!

**Break It Down**: Big tasks, like preparing for a test, feel easier when you split them into smaller steps. Study a little every day instead of cramming.

### Study Techniques

Try these methods to make studying stick:

**Active Recall**: Don't just read—test yourself! Close your book and see how much you can remember.

**Spaced Repetition**: Review what you've learned over time. For example, go over a topic today, then again in three days, and a week later.

**Mind Mapping**: Draw diagrams to connect ideas. It's like creating a map for your brain!

### Balanced Lifestyle

Your brain needs more than just books to work well:

**Move Around**: Play sports, dance, or take a walk. It boosts your focus and beats stress.

**Eat Healthy**: Snack on fruits, nuts, or yogurt instead of junk food to keep your energy up.

**Sleep Tight**: Get 8-10 hours of sleep every night. A rested mind learns better.

## Subject-Specific Tips

### Mathematics

Math can be fun if you practice it right:

**Solve Problems Daily**: The more you practice, the easier it gets.

**Understand, Don't Memorize**: Know why a formula works, not just what it is.

**Ask for Help**: Stuck on a problem? Ask your teacher or check online tutorials.

### Science

Make science come alive:

**Try Experiments**: Mix things at home (safely!) to see how concepts work.

**Watch Videos**: Visuals can help you understand tough topics like gravity or electricity.

**Connect to Life**: Think about how science explains rain, plants, or even your phone!

### Languages

Get better at Hindi, English, or any language:

**Read More**: Pick up storybooks, comics, or newspapers.

**Write Often**: Jot down your thoughts or short stories.

**Speak Up**: Chat with friends or family to practice speaking confidently.

## Overcoming Exam Pressure

Exams don't have to scare you. Here's how to stay calm and do your best:

**Start Early**: Study a little every day instead of rushing at the end.

**Relax**: Take deep breaths or tell yourself, "I've got this!" when you feel nervous.

**Plan Your Exam Time**: Don't spend too long on one question—keep moving.

## Resources for Students

Boost your learning with these tools:

**Websites**: Try Khan Academy, BYJU'S, or Toppr for lessons and quizzes.

**Apps**: Use Quizlet for flashcards or Duolingo for languages.

**Books**: Stick to your NCERT textbooks—they're perfect for the Indian curriculum.

## Conclusion

Learning in middle school doesn't have to be hard. By managing your time, using smart study tricks, staying healthy, and preparing for exams, you can do great in classes 6-10. It's all about taking small steps and believing in yourself.

So, grab your books, try these tips, and watch how you grow smarter every day. You've got the power to succeed—go for it!`,
    featured_image: 'https://images.pexels.com/photos/159711/books-bookstore-book-reading-159711.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1',
    author: {
      name: 'Evater Team',
      avatar: 'https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=150&h=150&dpr=1',
      bio: 'Educational experts dedicated to helping students succeed in their academic journey.'
    },
    category: 'Study Tips',
    tags: ['Middle School', 'Study Tips', 'Indian Education', 'Class 6-10', 'Learning Strategies', 'Time Management'],
    published_date: '2024-01-15',
    read_time: 8,
    is_featured: true,
    status: 'published',
    seo: {
      meta_title: 'How Students in Middle School Should Learn - Evater Blog',
      meta_description: 'Complete guide for Indian middle school students (classes 6-10) with effective learning strategies, study tips, and exam preparation techniques.'
    }
  }
]

// Helper functions for blog data
export const getFeaturedPosts = (): BlogPost[] => {
  const posts = blogPosts.filter(post => post.is_featured && post.status === 'published')
  console.log('getFeaturedPosts called, returning:', posts)
  return posts
}

export const getRecentPosts = (limit: number = 6): BlogPost[] => {
  const posts = blogPosts
    .filter(post => post.status === 'published')
    .sort((a, b) => new Date(b.published_date).getTime() - new Date(a.published_date).getTime())
    .slice(0, limit)
  console.log('getRecentPosts called, returning:', posts)
  return posts
}

export const getPostBySlug = (slug: string): BlogPost | undefined => {
  console.log('getPostBySlug called with slug:', slug)
  const post = blogPosts.find(post => post.slug === slug && post.status === 'published')
  console.log('getPostBySlug found post:', post)
  return post
}

export const getPostsByCategory = (category: string): BlogPost[] => {
  const posts = blogPosts.filter(post => 
    post.category === category && post.status === 'published'
  )
  console.log('getPostsByCategory called with category:', category, 'returning:', posts)
  return posts
}

export const getRelatedPosts = (currentPost: BlogPost, limit: number = 3): BlogPost[] => {
  const posts = blogPosts
    .filter(post => 
      post.id !== currentPost.id && 
      post.status === 'published' &&
      (post.category === currentPost.category || 
       post.tags.some(tag => currentPost.tags.includes(tag)))
    )
    .slice(0, limit)
  console.log('getRelatedPosts called for post:', currentPost.id, 'returning:', posts)
  return posts
}

export const searchPosts = (query: string): BlogPost[] => {
  const lowercaseQuery = query.toLowerCase()
  const posts = blogPosts.filter(post =>
    post.status === 'published' &&
    (post.title.toLowerCase().includes(lowercaseQuery) ||
     post.excerpt.toLowerCase().includes(lowercaseQuery) ||
     post.content.toLowerCase().includes(lowercaseQuery) ||
     post.tags.some(tag => tag.toLowerCase().includes(lowercaseQuery)))
  )
  console.log('searchPosts called with query:', query, 'returning:', posts)
  return posts
}