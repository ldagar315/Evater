import React from "react";
import { useNavigate } from "react-router-dom";
import { Calendar, Clock, ArrowRight } from "lucide-react";
import { Card, CardContent } from "../ui/Card";
import { BlogPost } from "../../types/blog";

interface BlogCardProps {
  post: BlogPost;
  featured?: boolean;
}

export function BlogCard({ post, featured = false }: BlogCardProps) {
  const navigate = useNavigate();

  const handleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigate(`/blog/${post.slug}`);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  return (
    <Card
      className={`group cursor-pointer overflow-hidden hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2 border border-neutral-100 hover:border-primary-200 h-full flex flex-col ${
        featured ? "md:col-span-2 lg:col-span-3 md:flex-row md:h-auto" : ""
      }`}
      onClick={handleClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          handleClick(e as any);
        }
      }}
    >
      <div
        className={`relative overflow-hidden ${
          featured ? "md:w-1/2 h-64 md:h-auto" : "h-56 w-full"
        }`}
      >
        <img
          src={post.featured_image}
          alt={post.title}
          className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
          onError={(e) => {
            e.currentTarget.src =
              "https://images.pexels.com/photos/159711/books-bookstore-book-reading-159711.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1";
          }}
        />
        <div className="absolute top-4 left-4">
          <span className="bg-white/90 backdrop-blur-sm text-primary-700 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider shadow-sm">
            {post.category}
          </span>
        </div>
      </div>

      <CardContent
        className={`flex flex-col flex-1 p-6 ${
          featured ? "md:w-1/2 justify-center p-8 lg:p-12" : ""
        }`}
      >
        <div className="flex items-center text-xs text-neutral-500 mb-3 space-x-3">
          <div className="flex items-center">
            <Calendar className="h-3 w-3 mr-1" />
            {formatDate(post.published_date)}
          </div>
          <div className="w-1 h-1 rounded-full bg-neutral-300"></div>
          <div className="flex items-center">
            <Clock className="h-3 w-3 mr-1" />
            {post.read_time} min read
          </div>
        </div>

        <h3
          className={`font-bold text-dark mb-3 group-hover:text-primary-600 transition-colors duration-200 leading-tight ${
            featured ? "text-2xl md:text-3xl" : "text-xl"
          }`}
        >
          {post.title}
        </h3>

        <p
          className={`text-neutral-600 mb-6 line-clamp-3 leading-relaxed ${
            featured ? "text-lg" : "text-sm"
          }`}
        >
          {post.excerpt}
        </p>

        <div className="mt-auto flex items-center justify-between">
          <div className="flex items-center">
            <img
              src={
                post.author.avatar ||
                "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=50&h=50&dpr=1"
              }
              alt={post.author.name}
              className="w-8 h-8 rounded-full mr-3 border border-neutral-200"
              onError={(e) => {
                e.currentTarget.src =
                  "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=50&h=50&dpr=1";
              }}
            />
            <span className="text-sm font-medium text-dark">
              {post.author.name}
            </span>
          </div>

          <div className="flex items-center text-primary-600 group-hover:text-primary-700 font-bold text-sm">
            <span className="mr-1">Read Article</span>
            <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform duration-200" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
