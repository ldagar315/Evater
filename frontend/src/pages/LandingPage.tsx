import React from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowRight,
  Sparkles,
  Target,
  Brain,
  CheckCircle,
  Star,
  Users,
  Award,
  BookOpen,
  Zap,
  Quote,
} from "lucide-react";
import { Button } from "../components/ui/Button";
import { Card, CardContent } from "../components/ui/Card";
import { Header } from "../components/layout/Header";
import { BlogSection } from "../components/blog/BlogSection";

export function LandingPage() {
  const navigate = useNavigate();

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Test Generation",
      description:
        "Create intelligent tests tailored to any subject, grade level, and difficulty with our advanced AI algorithms.",
      color: "bg-primary-50 text-primary-600",
    },
    {
      icon: Target,
      title: "AI-Powered Evaluation",
      description:
        "Upload handwritten answer sheets and get instant, accurate scoring with detailed feedback using cutting-edge AI technology.",
      color: "bg-secondary-50 text-secondary-600",
    },
    {
      icon: Sparkles,
      title: "Smart Feedback System",
      description:
        "Receive comprehensive insights, error analysis, and personalized improvement suggestions for every question.",
      color: "bg-purple-50 text-purple-600",
    },
  ];

  const stats = [
    { number: "10,000+", label: "Tests Generated", icon: BookOpen },
    { number: "50,000+", label: "Students Helped", icon: Users },
    { number: "95%", label: "Accuracy Rate", icon: Target },
    { number: "4.9/5", label: "User Rating", icon: Star },
  ];

  const benefits = [
    "Generate tests in seconds, not hours",
    "Support for all subjects and grade levels",
    "Automatic handwriting recognition",
    "Detailed performance analytics",
    "Personalized learning recommendations",
    "Export tests in multiple formats",
  ];

  const testimonials = [
    {
      name: "Darsh",
      grade: "Class 10th",
      school: "Presidium Sector-57",
      content:
        "Evater has completely transformed how I prepare for my board exams. The AI-generated tests are exactly what I need to practice, and the instant feedback helps me understand my mistakes immediately.",
      rating: 5,
      image: "/IMG-20251020-WA0056.jpg",
    },
    {
      name: "Manvi",
      grade: "Class 7th",
      school: "Presidium Sec-57",
      content:
        "I love how Evater makes studying fun! The tests are challenging but fair, and I can see my progress improving every day. My teachers are impressed with my performance.",
      rating: 5,
      image: "/IMG-20230912-WA0025.jpg",
    },
    {
      name: "Ishika",
      grade: "Class 10th",
      school: "Euro International Sector-45",
      content:
        "The AI checking feature is amazing! I can upload my handwritten answers and get detailed feedback instantly. It's like having a personal tutor available 24/7.",
      rating: 5,
      image: "/IMG_20241028_145336.jpg",
    },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-cream font-sans">
      <div className="flex-1">
        {/* Navigation */}
        <Header />

        {/* Hero Section */}
        <section className="relative py-20 lg:py-32 overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-primary-50 via-cream to-secondary-50 opacity-70"></div>
          <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
              <div className="animate-fade-in">
                <img
                  src="/Evater_logo_2.png"
                  alt="Evater Logo"
                  className="h-40 md:h-56 w-auto object-contain mx-auto mb-8 drop-shadow-sm"
                />
              </div>

              <h1 className="text-5xl lg:text-7xl font-bold text-dark mb-8 mt-8 leading-tight tracking-tight animate-slide-up">
                The Future of
                <span className="block bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent pb-2">
                  Test Creation
                </span>
                <span className="block">& Evaluation</span>
              </h1>

              <p
                className="text-xl lg:text-2xl text-neutral-600 mb-12 max-w-4xl mx-auto leading-relaxed font-medium animate-slide-up"
                style={{ animationDelay: "0.1s" }}
              >
                Transform your educational experience with AI-powered test
                generation and intelligent answer sheet evaluation. Create,
                evaluate, and improve learning outcomes in minutes, not hours.
              </p>

              <div
                className="flex flex-col sm:flex-row gap-6 justify-center items-center animate-slide-up"
                style={{ animationDelay: "0.2s" }}
              >
                <Button
                  onClick={() => navigate("/auth")}
                  size="lg"
                  className="text-lg px-8 py-4 shadow-lg shadow-primary-200 hover:shadow-xl transform hover:-translate-y-1 transition-all duration-200 rounded-xl bg-dark text-white hover:bg-neutral-800"
                >
                  Get Started Free
                  <ArrowRight className="h-5 w-5 ml-2" />
                </Button>
                <Button
                  variant="outline"
                  size="lg"
                  className="text-lg px-8 py-4 border-2 border-neutral-200 text-dark hover:bg-neutral-50 hover:border-neutral-300 rounded-xl"
                  onClick={() =>
                    document
                      .getElementById("features")
                      ?.scrollIntoView({ behavior: "smooth" })
                  }
                >
                  Learn More
                </Button>
              </div>
            </div>
          </div>
        </section>

        {/* Stats Section */}
        <section className="py-16 bg-white border-y border-neutral-100">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="bg-white rounded-2xl shadow-sm border border-neutral-100 p-8 lg:p-12">
              <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 divide-y md:divide-y-0 md:divide-x divide-neutral-100">
                {stats.map((stat, index) => {
                  const Icon = stat.icon;
                  return (
                    <div key={index} className="text-center pt-8 md:pt-0 px-4">
                      <div className="flex justify-center mb-4">
                        <div className="w-14 h-14 bg-primary-50 rounded-2xl flex items-center justify-center transform transition-transform hover:scale-110 duration-300">
                          <Icon className="h-7 w-7 text-primary-600" />
                        </div>
                      </div>
                      <div className="text-3xl lg:text-4xl font-bold text-dark mb-2 tracking-tight">
                        {stat.number}
                      </div>
                      <div className="text-neutral-500 font-semibold uppercase tracking-wider text-sm">
                        {stat.label}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </section>

        {/* Features Section */}
        <section id="features" className="py-24 bg-cream">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-16">
              <span className="text-primary-600 font-bold tracking-wider uppercase text-sm mb-3 block">
                Features
              </span>
              <h2 className="text-4xl lg:text-5xl font-bold text-dark mb-6 tracking-tight">
                Revolutionize Your Teaching
              </h2>
              <p className="text-xl text-neutral-600 max-w-3xl mx-auto leading-relaxed">
                Experience the power of AI-driven education technology that
                adapts to your needs and enhances learning outcomes.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {features.map((feature, index) => {
                const Icon = feature.icon;
                return (
                  <Card
                    key={index}
                    className="group relative bg-white rounded-2xl p-8 border border-neutral-100 shadow-sm transition-all duration-300 hover:shadow-xl hover:-translate-y-2 cursor-pointer overflow-hidden"
                  >
                    <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-transparent to-neutral-50 rounded-bl-full opacity-50 transition-opacity group-hover:opacity-100"></div>
                    <CardContent className="p-0 text-center relative z-10">
                      <div className="flex justify-center mb-8">
                        <div
                          className={`w-20 h-20 rounded-2xl flex items-center justify-center ${feature.color} shadow-sm group-hover:scale-110 transition-transform duration-300`}
                        >
                          <Icon className="h-10 w-10" />
                        </div>
                      </div>
                      <h3 className="text-2xl font-bold text-dark mb-4">
                        {feature.title}
                      </h3>
                      <p className="text-neutral-600 leading-relaxed text-lg">
                        {feature.description}
                      </p>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          </div>
        </section>

        {/* Testimonials Section */}
        <section className="py-24 bg-white relative overflow-hidden">
          <div className="absolute top-0 left-0 w-full h-full bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-[0.03]"></div>
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
            <div className="text-center mb-16">
              <span className="text-secondary-600 font-bold tracking-wider uppercase text-sm mb-3 block">
                Testimonials
              </span>
              <h2 className="text-4xl lg:text-5xl font-bold text-dark mb-6 tracking-tight">
                What Students Say
              </h2>
              <p className="text-xl text-neutral-600 max-w-3xl mx-auto leading-relaxed">
                Hear from students who have transformed their learning
                experience with Evater
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {testimonials.map((testimonial, index) => (
                <Card
                  key={index}
                  className="group bg-cream/50 hover:bg-white rounded-2xl border border-neutral-100 hover:border-secondary-200 shadow-sm hover:shadow-xl transition-all duration-300 transform hover:-translate-y-2"
                >
                  <CardContent className="p-8">
                    <div className="flex justify-center mb-8">
                      <div className="w-24 h-24 rounded-full p-1 bg-gradient-to-br from-secondary-200 to-primary-200">
                        <div className="w-full h-full rounded-full overflow-hidden border-4 border-white">
                          <img
                            src={testimonial.image}
                            alt={testimonial.name}
                            className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                            onError={(e) => {
                              // Fallback to a default avatar if image fails to load
                              e.currentTarget.src =
                                "https://images.pexels.com/photos/774909/pexels-photo-774909.jpeg?auto=compress&cs=tinysrgb&w=300&h=300&dpr=1";
                            }}
                          />
                        </div>
                      </div>
                    </div>

                    <div className="flex justify-center mb-6">
                      {[...Array(testimonial.rating)].map((_, i) => (
                        <Star
                          key={i}
                          className="h-5 w-5 text-secondary-400 fill-current"
                        />
                      ))}
                    </div>

                    <div className="relative mb-8">
                      <Quote className="h-8 w-8 text-secondary-200 absolute -top-4 -left-2 transform -scale-x-100 opacity-50" />
                      <p className="text-neutral-700 leading-relaxed italic text-center relative z-10 px-4">
                        "{testimonial.content}"
                      </p>
                      <Quote className="h-8 w-8 text-secondary-200 absolute -bottom-4 -right-2 opacity-50" />
                    </div>

                    <div className="text-center border-t border-neutral-200/50 pt-6">
                      <h4 className="text-xl font-bold text-dark mb-1">
                        {testimonial.name}
                      </h4>
                      <p className="text-sm text-primary-600 font-bold uppercase tracking-wide mb-1">
                        {testimonial.grade}
                      </p>
                      <p className="text-sm text-neutral-500 font-medium">
                        {testimonial.school}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </section>

        {/* Blog Section */}
        <BlogSection />

        {/* Benefits Section */}
        <section className="py-24 bg-cream overflow-hidden">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
              <div className="order-2 lg:order-1">
                <h2 className="text-4xl lg:text-5xl font-bold text-dark mb-8 tracking-tight">
                  Why Choose Evater?
                </h2>
                <p className="text-xl text-neutral-600 mb-10 leading-relaxed">
                  Join thousands of educators who have transformed their
                  teaching experience with our cutting-edge platform.
                </p>

                <div className="space-y-6">
                  {benefits.map((benefit, index) => (
                    <div key={index} className="flex items-center group">
                      <div className="w-8 h-8 bg-primary-100 rounded-full flex items-center justify-center mr-4 flex-shrink-0 group-hover:bg-primary-500 transition-colors duration-300">
                        <CheckCircle className="h-5 w-5 text-primary-600 group-hover:text-white transition-colors duration-300" />
                      </div>
                      <span className="text-lg text-neutral-700 font-medium group-hover:text-dark transition-colors">
                        {benefit}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="relative order-1 lg:order-2">
                <div className="absolute -inset-4 bg-gradient-to-r from-primary-200 to-secondary-200 rounded-[2.5rem] blur-2xl opacity-30 animate-pulse"></div>
                <div className="relative bg-gradient-to-br from-dark to-neutral-900 rounded-[2rem] p-10 text-white shadow-2xl transform rotate-2 hover:rotate-0 transition-transform duration-500 border border-white/10">
                  <div className="absolute top-0 right-0 p-8 opacity-20">
                    <Award className="h-32 w-32 text-white" />
                  </div>

                  <div className="relative z-10">
                    <div className="inline-flex items-center bg-white/10 backdrop-blur-md rounded-full px-4 py-1.5 text-sm font-bold text-primary-300 mb-6 border border-white/10">
                      <Sparkles className="h-4 w-4 mr-2" />
                      Premium Experience
                    </div>

                    <h3 className="text-3xl font-bold mb-6">
                      Unlock Full Potential
                    </h3>

                    <div className="bg-white/5 backdrop-blur-sm rounded-2xl p-6 mb-8 border border-white/10">
                      <ul className="space-y-4">
                        {[
                          "Unlimited test generation",
                          "Advanced analytics dashboard",
                          "Priority customer support",
                          "Custom branding options",
                        ].map((item, i) => (
                          <li
                            key={i}
                            className="flex items-center text-lg text-white/90"
                          >
                            <div className="w-1.5 h-1.5 rounded-full bg-primary-400 mr-3"></div>
                            {item}
                          </li>
                        ))}
                      </ul>
                    </div>

                    <Button
                      variant="secondary"
                      size="lg"
                      className="w-full text-dark font-bold text-lg py-4 rounded-xl shadow-lg shadow-secondary-500/20 hover:shadow-secondary-500/40"
                      onClick={() => navigate("/auth")}
                    >
                      Start Your Free Trial
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* CTA Section */}
        <section className="py-24 bg-gradient-to-br from-primary-600 via-primary-700 to-secondary-700 relative overflow-hidden">
          <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-10"></div>
          <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-black/10 to-transparent"></div>

          <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center relative z-10">
            <h2 className="text-4xl lg:text-6xl font-bold text-white mb-8 tracking-tight">
              Ready to Transform Your Teaching?
            </h2>
            <p className="text-xl lg:text-2xl text-white/90 mb-12 leading-relaxed max-w-3xl mx-auto font-medium">
              Join the educational revolution and experience the future of test
              creation and evaluation today.
            </p>

            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
              <Button
                onClick={() => navigate("/auth")}
                variant="secondary"
                size="lg"
                className="text-lg px-10 py-5 text-dark font-bold shadow-2xl shadow-black/20 hover:shadow-black/30 transform hover:scale-105 transition-all duration-200 rounded-xl min-w-[200px]"
              >
                Get Started Now
                <ArrowRight className="h-5 w-5 ml-2" />
              </Button>
              <p className="text-white/80 text-sm font-medium bg-white/10 backdrop-blur px-4 py-2 rounded-full border border-white/10">
                No credit card required • Free forever plan available
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
