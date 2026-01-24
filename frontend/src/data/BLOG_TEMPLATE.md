# Blog Template - Adding New Blog Posts

This guide explains how to add new blog posts to the Evater blog system.

## Quick Start

1. Open `src/data/blogPosts.ts`
2. Add a new blog post object to the `blogPosts` array
3. Follow the template structure below
4. Save the file - your blog will automatically appear!

## Blog Post Template

```typescript
{
  id: '2', // Unique ID - increment from the last blog post
  title: 'Your Blog Post Title',
  slug: 'your-blog-post-title', // URL-friendly version of title (lowercase, hyphens)
  excerpt: 'A brief summary of your blog post that appears on the blog listing page. Keep it under 200 characters for best display.',
  content: `# Your Blog Post Title

Your full blog content goes here in markdown format.

## Section Header

You can use markdown formatting:

**Bold text**

- Bullet points
- Another bullet point

### Subsection

Regular paragraphs of text go here.

## Another Section

More content...

## Conclusion

Wrap up your blog post here.`,
  featured_image: 'https://images.pexels.com/photos/PHOTO_ID/...', // URL to blog hero image
  author: {
    name: 'Evater Team',
    avatar: 'https://images.pexels.com/photos/AVATAR_ID/...',
    bio: 'Educational experts dedicated to helping students succeed in their academic journey.'
  },
  category: 'Study Tips', // Choose from existing categories or create new one
  tags: ['Tag1', 'Tag2', 'Tag3'], // Array of relevant tags for filtering
  published_date: '2024-01-20', // Format: YYYY-MM-DD
  read_time: 5, // Estimated reading time in minutes
  is_featured: false, // Set to true for featured blog posts
  status: 'published', // 'published' or 'draft'
  seo: {
    meta_title: 'Your Blog Post Title - Evater Blog',
    meta_description: 'SEO-friendly description for search engines. Keep under 160 characters.'
  }
}
```

## Field Explanations

### Required Fields

- **id**: Unique identifier (string). Use incrementing numbers: '1', '2', '3', etc.
- **title**: The blog post headline shown to users
- **slug**: URL-friendly version of the title (no spaces, all lowercase, hyphens only)
  - Example: "How to Study Better" → "how-to-study-better"
- **excerpt**: Short summary (1-2 sentences) shown on blog listing page
- **content**: Full blog content in markdown format
- **featured_image**: URL to a high-quality image (recommended: 1260x750px)
- **author**: Object containing author details
- **category**: Category name (string)
- **tags**: Array of relevant keywords for the post
- **published_date**: Publication date in YYYY-MM-DD format
- **read_time**: Estimated reading time in minutes
- **status**: Either 'published' or 'draft'

### Optional Fields

- **is_featured**: Set to `true` for featured posts (shows with special styling)
- **seo**: Object containing SEO metadata
  - **meta_title**: Custom title for search engines
  - **meta_description**: Description for search results

## Content Formatting Guide

The `content` field supports basic markdown:

- `# Heading 1` - Main section headers
- `## Heading 2` - Subsection headers
- `### Heading 3` - Sub-subsection headers
- `**Bold Text**` - Bold formatting (for full paragraph)
- `- List item` - Bullet points
- Empty lines create paragraph breaks

## Tips for Great Blog Posts

1. **Compelling Title**: Make it clear, specific, and engaging
2. **Strong Excerpt**: Hook readers in 1-2 sentences
3. **Quality Images**: Use high-resolution images from Pexels or similar
4. **Good Structure**: Use headings to organize content
5. **SEO Optimization**: Include relevant keywords in title, excerpt, and content
6. **Appropriate Tags**: Add 4-6 relevant tags for better discoverability
7. **Accurate Read Time**: Estimate ~200 words per minute

## Finding Images

Good sources for free, high-quality images:

- [Pexels](https://www.pexels.com/)
- [Unsplash](https://unsplash.com/)
- [Pixabay](https://pixabay.com/)

Use the direct image URL from these services.

## Example Categories

- Study Tips
- EdTech
- Test Preparation
- Learning Strategies
- Student Success
- Teaching Methods

## Example Tags

- Middle School
- High School
- Indian Education
- CBSE
- Time Management
- Exam Tips
- Study Techniques
- Online Learning

## Testing Your Blog Post

After adding your blog post:

1. Save the file
2. Navigate to `/blog` in your browser
3. Check that your blog appears in the listing
4. Click on your blog to verify:
   - Content displays correctly
   - Images load properly
   - Date and metadata are correct
   - Related posts appear (if applicable)

## Troubleshooting

**Blog doesn't appear?**

- Check that `status` is set to `'published'`
- Verify the `blogPosts` array syntax is correct (commas, brackets)
- Check browser console for errors

**Images not loading?**

- Verify image URLs are correct and accessible
- Ensure URLs start with `https://`
- Test the URL in a browser

**Formatting looks wrong?**

- Check markdown syntax
- Ensure content is wrapped in backticks: \`content here\`
- Verify line breaks are preserved

---

## Future Enhancement: Modular Structure

In the future, we can restructure blogs into individual files in a `src/data/blogs/` directory. This will make it easier to manage many blog posts and avoid merge conflicts when multiple people add content.
