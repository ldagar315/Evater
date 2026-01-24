import React from "react";
import { useNavigate } from "react-router-dom";
import { Target, MessageSquare, Sparkles } from "lucide-react";
import { Header } from "../components/layout/Header";
import { Button } from "../components/ui/Button";

const missionHighlights = [
  {
    title: "Learning without feedback",
    description:
      "Students can stream lessons, download notes, or ask an AI to explain anything, yet meaningful feedback on answers is still rare.",
    icon: MessageSquare,
    accent: "bg-primary-50 text-primary-600",
  },
  {
    title: "Evater's promise",
    description:
      "We are building the first feedback-focused learning platform where every test, answer sheet, and viva produces guided critique.",
    icon: Sparkles,
    accent: "bg-secondary-50 text-secondary-600",
  },
  {
    title: "Better feedback, best learning",
    description:
      "When every learner knows exactly what to fix, practice compounds faster. A teacher to every student becomes a reality.",
    icon: Target,
    accent: "bg-purple-50 text-purple-600",
  },
];

const founderStats = [
  { label: "Role", value: "Founder & Builder" },
  { label: "Education", value: "IIT Delhi - Class of 2024" },
  { label: "Focus", value: "Building steadily with care" },
];

export function AboutPage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-cream font-sans">
      <Header />
      <main className="max-w-7xl mx-auto py-16 px-4 sm:px-6 lg:px-8">
        <section className="text-center mb-16 animate-fade-in">
          <span className="inline-flex items-center px-4 py-1 rounded-full bg-primary-50 text-primary-600 text-sm font-semibold tracking-wide uppercase mb-6">
            About Evater
          </span>
          <h1 className="text-4xl md:text-5xl font-bold text-dark tracking-tight mb-6">
            A teacher to every student
          </h1>
          <p className="text-xl text-neutral-500 leading-relaxed max-w-3xl mx-auto mb-10">
            Evater exists to close the loop between learning and mastery. We
            obsess over feedback so every student knows what to improve, why it
            matters, and what to do next.
          </p>
          <Button
            onClick={() => navigate("/auth")}
            className="inline-flex items-center px-8 py-3 shadow-lg shadow-primary-500/10"
          >
            Experience Evater
          </Button>
        </section>

        <section className="bg-white rounded-3xl border border-neutral-100 shadow-sm p-8 md:p-12 mb-16 animate-slide-up">
          <div className="mb-10">
            <p className="text-sm font-semibold tracking-wider text-primary-500 uppercase mb-4">
              Our Mission
            </p>
            <h2 className="text-3xl md:text-4xl font-bold text-dark mb-6">
              A feedback focused first learning platform
            </h2>
            <p className="text-lg text-neutral-600 leading-relaxed">
              Today there are countless avenues to learn - videos, notes, AI
              tutors, and endless repositories of questions. But almost none of
              them stay with you after the attempt to explain what happened.
              Evater is changing that by weaving deep, actionable feedback into
              every interaction. Because better feedback unlocks the best
              learning outcomes.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {missionHighlights.map((item) => (
              <div
                key={item.title}
                className="p-6 rounded-2xl border border-neutral-100 bg-cream/60 hover:-translate-y-1 transition-all duration-200"
              >
                <div
                  className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${item.accent}`}
                >
                  <item.icon className="w-6 h-6" />
                </div>
                <h3 className="text-lg font-semibold text-dark mb-2">
                  {item.title}
                </h3>
                <p className="text-sm text-neutral-500 leading-relaxed">
                  {item.description}
                </p>
              </div>
            ))}
          </div>
        </section>

        <div className="bg-primary-50 rounded-3xl border border-primary-100 p-8 md:p-10 text-center mb-16">
          <p className="text-primary-700 text-lg font-semibold">
            Better feedback. Best learning. That is how we bring a teacher to
            every student.
          </p>
        </div>

        <section className="bg-white rounded-3xl border border-neutral-100 shadow-sm p-8 md:p-12 animate-slide-up">
          <p className="text-sm font-semibold tracking-wider text-primary-500 uppercase mb-4">
            Meet the Team
          </p>
          <h2 className="text-3xl md:text-4xl font-bold text-dark mb-6">
            About Lakshay Dagar
          </h2>
          <div className="flex flex-col lg:flex-row gap-8">
            <div className="w-full lg:w-1/3 flex justify-center">
              <div className="w-32 h-32 rounded-3xl bg-primary-50 border border-primary-100 flex items-center justify-center text-3xl font-bold text-primary-600">
                LD
              </div>
            </div>
            <div className="w-full lg:w-2/3">
              <p className="text-lg text-neutral-600 leading-relaxed mb-6">
                I&apos;m Lakshay Dagar, a 2024 graduate from IIT Delhi. This gap
                between "being taught" and "being truly guided" fascinated me, so
                I&apos;m building Evater solely, slowly, and steadily. My focus
                is to craft a system where every learner receives the kind of
                thoughtful feedback a great teacher would give.
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                {founderStats.map((stat) => (
                  <div
                    key={stat.label}
                    className="p-4 rounded-2xl bg-cream/70 border border-neutral-100"
                  >
                    <p className="text-sm font-semibold text-neutral-500 uppercase tracking-wide mb-1">
                      {stat.label}
                    </p>
                    <p className="text-dark font-semibold">{stat.value}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
