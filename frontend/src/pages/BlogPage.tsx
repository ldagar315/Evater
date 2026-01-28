import React, { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { BookOpen, Search } from "lucide-react";
import { Header } from "../components/layout/Header";
import { Footer } from "../components/layout/Footer";
import { BlogCard } from "../components/blog/BlogCard";
import { Card, CardContent } from "../components/ui/Card";
import { useAuthContext } from "../contexts/AuthContext";
import { blogPosts } from "../data/blogPosts";

export function BlogPage() {
  const navigate = useNavigate();
  const { user } = useAuthContext();
  const [currentPage, setCurrentPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState("");
  const postsPerPage = 9;

  // Get all published posts
  const filteredPosts = useMemo(() => {
    let posts = blogPosts.filter((post) => post.status === "published");

    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      posts = posts.filter(
        (post) =>
          post.title.toLowerCase().includes(query) ||
          post.excerpt.toLowerCase().includes(query) ||
          post.category.toLowerCase().includes(query)
      );
    }

    // Sort by date (newest first)
    return posts.sort(
      (a, b) =>
        new Date(b.published_date).getTime() -
        new Date(a.published_date).getTime()
    );
  }, [searchQuery]);

  // Pagination
  const totalPages = Math.ceil(filteredPosts.length / postsPerPage);
  const startIndex = (currentPage - 1) * postsPerPage;
  const paginatedPosts = filteredPosts.slice(
    startIndex,
    startIndex + postsPerPage
  );

  return (
    <div className="min-h-screen flex flex-col bg-cream font-sans">
      <Header />

      <div className="flex-1">
        {/* Hero Section */}
        <section className="relative bg-dark py-20 lg:py-32 overflow-hidden">
          <div className="absolute inset-0 opacity-10">
            <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')]"></div>
          </div>
          <div className="absolute inset-0 bg-gradient-to-br from-primary-900/20 to-secondary-900/20"></div>

          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
            <span className="inline-block py-1 px-3 rounded-full bg-primary-500/20 text-primary-300 text-sm font-bold tracking-wider uppercase mb-6 border border-primary-500/30">
              Evater Blog
            </span>
            <h1 className="text-4xl lg:text-6xl font-bold text-white mb-6 leading-tight tracking-tight">
              Insights for{" "}
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-secondary-400">
                Better Learning
              </span>
            </h1>

            <p className="text-xl text-neutral-400 mb-10 max-w-2xl mx-auto leading-relaxed">
              Discover the latest trends in edTech, study techniques, and how AI
              is transforming education.
            </p>

            <div className="max-w-md mx-auto relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-neutral-500" />
              </div>
              <input
                type="text"
                placeholder="Search articles..."
                className="block w-full pl-11 pr-4 py-4 bg-white/10 border border-white/20 rounded-xl text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent backdrop-blur-sm transition-all"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
          </div>
        </section>

        {/* Blog Posts Grid */}
        <section className="py-16 lg:py-24 bg-cream relative">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            {/* Featured Post (First post if on page 1 and no search) */}
            {currentPage === 1 && !searchQuery && paginatedPosts.length > 0 && (
              <div className="mb-16">
                <h2 className="text-2xl font-bold text-dark mb-8 flex items-center">
                  <span className="w-2 h-8 bg-primary-500 rounded-full mr-3"></span>
                  Featured Article
                </h2>
                <BlogCard post={paginatedPosts[0]} featured={true} />
              </div>
            )}

            {/* Remaining Posts */}
            <div className="mb-12">
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-2xl font-bold text-dark flex items-center">
                  <span className="w-2 h-8 bg-secondary-500 rounded-full mr-3"></span>
                  {searchQuery ? "Search Results" : "Latest Articles"}
                </h2>
                <p className="text-neutral-500 text-sm">
                  Showing {paginatedPosts.length} of {filteredPosts.length}{" "}
                  articles
                </p>
              </div>

              {paginatedPosts.length === 0 ? (
                <Card className="bg-white border-dashed border-2 border-neutral-200 shadow-none">
                  <CardContent className="p-16 text-center">
                    <BookOpen className="h-16 w-16 text-neutral-300 mx-auto mb-6" />
                    <h3 className="text-xl font-bold text-dark mb-2">
                      No Articles Found
                    </h3>
                    <p className="text-neutral-500 mb-6 max-w-md mx-auto">
                      We couldn't find any articles matching your search. Try
                      adjusting your keywords or browse all articles.
                    </p>
                    <button
                      onClick={() => setSearchQuery("")}
                      className="text-primary-600 font-semibold hover:text-primary-700"
                    >
                      Clear Search
                    </button>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                  {(currentPage === 1 && !searchQuery
                    ? paginatedPosts.slice(1)
                    : paginatedPosts
                  ).map((post) => (
                    <BlogCard key={post.id} post={post} />
                  ))}
                </div>
              )}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="flex justify-center items-center space-x-2 mt-16">
                <button
                  onClick={() =>
                    setCurrentPage((prev) => Math.max(prev - 1, 1))
                  }
                  disabled={currentPage === 1}
                  className="px-4 py-2 text-sm font-medium text-neutral-700 bg-white border border-neutral-200 rounded-lg hover:bg-neutral-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Previous
                </button>

                {Array.from({ length: totalPages }, (_, i) => i + 1).map(
                  (page) => (
                    <button
                      key={page}
                      onClick={() => setCurrentPage(page)}
                      className={`w-10 h-10 flex items-center justify-center text-sm font-bold rounded-lg transition-all ${
                        currentPage === page
                          ? "bg-primary-500 text-white shadow-md transform scale-105"
                          : "text-neutral-600 bg-white border border-neutral-200 hover:bg-neutral-50 hover:border-neutral-300"
                      }`}
                    >
                      {page}
                    </button>
                  )
                )}

                <button
                  onClick={() =>
                    setCurrentPage((prev) => Math.min(prev + 1, totalPages))
                  }
                  disabled={currentPage === totalPages}
                  className="px-4 py-2 text-sm font-medium text-neutral-700 bg-white border border-neutral-200 rounded-lg hover:bg-neutral-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  Next
                </button>
              </div>
            )}
          </div>
        </section>
      </div>
      <Footer />
    </div>
  );
}
