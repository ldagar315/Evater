import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Calendar,
  Clock,
  Share2,
  Tag,
  ChevronRight,
  Home,
} from "lucide-react";
import Markdown from "react-markdown";
import { Header } from "../components/layout/Header";
import { Footer } from "../components/layout/Footer";
import { BlogCard } from "../components/blog/BlogCard";
import { Button } from "../components/ui/Button";
import { Card, CardContent } from "../components/ui/Card";
import { useAuthContext } from "../contexts/AuthContext";
import { getPostBySlug, getRelatedPosts } from "../data/blogPosts";
import { BlogPost } from "../types/blog";

export function BlogPostPage() {
  const { slug } = useParams<{ slug: string }>();
  const navigate = useNavigate();
  const { user } = useAuthContext();
  const [post, setPost] = useState<BlogPost | null>(null);
  const [relatedPosts, setRelatedPosts] = useState<BlogPost[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (slug) {
      const foundPost = getPostBySlug(slug);

      if (foundPost) {
        setPost(foundPost);
        setRelatedPosts(getRelatedPosts(foundPost));

        // Update page title and meta description
        document.title =
          foundPost.seo?.meta_title || `${foundPost.title} - Evater Blog`;

        const metaDescription = document.querySelector(
          'meta[name="description"]'
        );
        if (metaDescription) {
          metaDescription.setAttribute(
            "content",
            foundPost.seo?.meta_description || foundPost.excerpt
          );
        }
      }
      setLoading(false);
    } else {
      setLoading(false);
    }
  }, [slug]);

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const handleShare = async () => {
    if (navigator.share && post) {
      try {
        await navigator.share({
          title: post.title,
          text: post.excerpt,
          url: window.location.href,
        });
      } catch (err) {
        navigator.clipboard.writeText(window.location.href);
      }
    } else {
      navigator.clipboard.writeText(window.location.href);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col bg-cream">
        <Header />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
            <p className="text-neutral-600">Loading article...</p>
          </div>
        </div>
        <Footer />
      </div>
    );
  }

  if (!post) {
    return (
      <div className="min-h-screen flex flex-col bg-cream">
        <Header />
        <div className="flex-1 flex items-center justify-center">
          <Card className="max-w-md w-full mx-4">
            <CardContent className="p-8 text-center">
              <h2 className="text-xl font-bold text-dark mb-2">
                Article Not Found
              </h2>
              <p className="text-neutral-600 mb-6">
                The article you're looking for doesn't exist or has been
                removed.
              </p>
              <Button onClick={() => navigate("/blog")}>Back to Blog</Button>
            </CardContent>
          </Card>
        </div>
        <Footer />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col bg-cream font-sans">
      <Header />

      <div className="flex-1">
        {/* Breadcrumb */}
        <div className="bg-white border-b border-neutral-100">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div className="flex items-center text-sm text-neutral-500">
              <Home
                className="h-4 w-4 mr-2 cursor-pointer hover:text-primary-600"
                onClick={() => navigate("/")}
              />
              <ChevronRight className="h-4 w-4 mx-2" />
              <span
                className="cursor-pointer hover:text-primary-600"
                onClick={() => navigate("/blog")}
              >
                Blog
              </span>
              <ChevronRight className="h-4 w-4 mx-2" />
              <span className="text-neutral-800 font-medium truncate max-w-[200px] sm:max-w-md">
                {post.title}
              </span>
            </div>
          </div>
        </div>

        {/* Hero Section */}
        <section className="relative py-20 lg:py-32 bg-dark overflow-hidden">
          <div className="absolute inset-0">
            <img
              src={post.featured_image}
              alt={post.title}
              className="w-full h-full object-cover opacity-20 blur-sm scale-105"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-dark via-dark/80 to-dark/40"></div>
          </div>

          <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <div className="mb-8">
              <span className="bg-primary-500 text-white px-4 py-1.5 rounded-full text-sm font-bold tracking-wide uppercase shadow-lg shadow-primary-500/20">
                {post.category}
              </span>
            </div>

            <h1 className="text-4xl lg:text-6xl font-bold text-white mb-8 leading-tight tracking-tight">
              {post.title}
            </h1>

            <div className="flex flex-wrap items-center justify-center gap-6 text-neutral-300 mb-8">
              <div className="flex items-center">
                <img
                  src={
                    post.author.avatar ||
                    "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=50&h=50&dpr=1"
                  }
                  alt={post.author.name}
                  className="w-10 h-10 rounded-full mr-3 border-2 border-white/10"
                />
                <div className="text-left">
                  <div className="font-bold text-white text-sm">
                    {post.author.name}
                  </div>
                  <div className="text-xs text-primary-400">Author</div>
                </div>
              </div>

              <div className="w-px h-8 bg-white/10 hidden sm:block"></div>

              <div className="flex items-center text-sm font-medium">
                <Calendar className="h-4 w-4 mr-2 text-primary-400" />
                {formatDate(post.published_date)}
              </div>

              <div className="flex items-center text-sm font-medium">
                <Clock className="h-4 w-4 mr-2 text-primary-400" />
                {post.read_time} min read
              </div>
            </div>
          </div>
        </section>

        {/* Article Content */}
        <section className="relative -mt-10 pb-20">
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="bg-white rounded-2xl shadow-xl border border-neutral-100 overflow-hidden">
              <div className="p-8 lg:p-12">
                <article className="prose prose-lg max-w-none prose-headings:font-bold prose-headings:text-dark prose-p:text-neutral-600 prose-p:leading-relaxed prose-a:text-primary-600 prose-a:no-underline hover:prose-a:underline prose-img:rounded-xl prose-strong:text-dark prose-blockquote:border-l-4 prose-blockquote:border-primary-500 prose-blockquote:bg-primary-50 prose-blockquote:py-2 prose-blockquote:px-4 prose-blockquote:not-italic prose-li:text-neutral-600">
                  <Markdown>{post.content}</Markdown>
                </article>

                {/* Tags */}
                {post.tags.length > 0 && (
                  <div className="mt-12 pt-8 border-t border-neutral-100">
                    <div className="flex items-center flex-wrap gap-2">
                      <Tag className="h-4 w-4 text-neutral-400 mr-2" />
                      {post.tags.map((tag, index) => (
                        <span
                          key={index}
                          className="bg-neutral-50 text-neutral-600 border border-neutral-200 px-3 py-1 rounded-full text-sm font-medium hover:bg-primary-50 hover:text-primary-700 hover:border-primary-200 transition-all duration-200 cursor-pointer"
                        >
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Share & Author */}
                <div className="mt-12 flex flex-col sm:flex-row items-center justify-between gap-6 pt-8 border-t border-neutral-100">
                  <div className="flex items-center space-x-4">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleShare}
                      className="text-neutral-600 hover:text-primary-600 hover:border-primary-200"
                    >
                      <Share2 className="h-4 w-4 mr-2" />
                      Share Article
                    </Button>
                  </div>

                  <div className="flex items-center">
                    <span className="text-neutral-500 text-sm mr-4">
                      Written by
                    </span>
                    <div className="flex items-center">
                      <img
                        src={
                          post.author.avatar ||
                          "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=50&h=50&dpr=1"
                        }
                        alt={post.author.name}
                        className="w-10 h-10 rounded-full mr-3"
                      />
                      <div>
                        <div className="font-bold text-dark text-sm">
                          {post.author.name}
                        </div>
                        <div className="text-xs text-neutral-500">
                          {post.author.bio}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Related Posts */}
        {relatedPosts.length > 0 && (
          <section className="py-16 bg-neutral-50 border-t border-neutral-200">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
              <div className="flex items-center justify-between mb-10">
                <h2 className="text-2xl font-bold text-dark">
                  Related Articles
                </h2>
                <Button
                  variant="ghost"
                  onClick={() => navigate("/blog")}
                  className="text-primary-600 hover:text-primary-700 hover:bg-primary-50"
                >
                  View All <ArrowLeft className="h-4 w-4 ml-2 rotate-180" />
                </Button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {relatedPosts.map((relatedPost) => (
                  <BlogCard key={relatedPost.id} post={relatedPost} />
                ))}
              </div>
            </div>
          </section>
        )}

        {/* CTA Section */}
        <section className="py-20 bg-gradient-to-r from-primary-600 to-secondary-600 relative overflow-hidden">
          <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
            <h2 className="text-3xl lg:text-4xl font-bold text-white mb-6">
              Ready to Transform Your Learning?
            </h2>
            <p className="text-xl text-white/90 mb-10 max-w-2xl mx-auto">
              Join thousands of teachers and students using Evater to create
              better assessments and improve learning outcomes.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                variant="secondary"
                size="lg"
                onClick={() => navigate(user ? "/home" : "/auth")}
                className="text-dark font-bold shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all"
              >
                {user ? "Go to Dashboard" : "Get Started Free"}
              </Button>
              <Button
                variant="outline"
                size="lg"
                onClick={() => navigate("/blog")}
                className="text-white border-white hover:bg-white hover:text-primary-600"
              >
                Read More Articles
              </Button>
            </div>
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
}
