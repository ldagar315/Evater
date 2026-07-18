import React from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Brain,
  Coins,
  FileText,
  GraduationCap,
  Heart,
  History,
  MessageSquare,
  PlusCircle,
  School,
  Sparkles,
  Target,
} from "lucide-react";
import { Button } from "../components/ui/Button";
import { Header } from "../components/layout/Header";
import { useAppState } from "../contexts/AppStateContext";
import { useAuthContext } from "../contexts/AuthContext";
import { useProfile } from "../hooks/useProfile";
import { BYPASS_AUTH } from "../lib/auth/devBypass";

function formatDate(date?: string) {
  if (!date) return "Recently";

  const parsedDate = new Date(date);
  if (Number.isNaN(parsedDate.getTime())) return "Recently";

  return parsedDate.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export function HomePage() {
  const navigate = useNavigate();
  const { appState } = useAppState();
  const { user } = useAuthContext();
  const { profile } = useProfile(user?.id);

  const firstName =
    profile?.user_name?.trim().split(/\s+/)[0] ||
    profile?.name?.trim().split(/\s+/)[0] ||
    "Educator";
  const lastTest = appState.last_generated_test;
  const lastFeedback = appState.last_generated_feedback;
  const latestTestTitle = lastTest
    ? [lastTest.subject, lastTest.chapter].filter(Boolean).join(" · ") ||
      "Generated test"
    : "No test generated yet";
  const latestFeedbackTitle = lastFeedback
    ? `Evaluation from ${formatDate(lastFeedback.created_at)}`
    : "No feedback available yet";

  const accountStats = [
    {
      label: "Grade",
      value: profile?.grade ? `Class ${profile.grade}` : "Not set",
      icon: GraduationCap,
      iconClass: "bg-primary-50 text-primary-600",
    },
    {
      label: "School",
      value: profile?.school || "Not set",
      icon: School,
      iconClass: "bg-secondary-50 text-secondary-700",
    },
    {
      label: "Credits",
      value: profile?.credits ?? "—",
      icon: Coins,
      iconClass: "bg-yellow-50 text-yellow-700",
    },
  ];

  const primaryActions = [
    {
      eyebrow: "Build",
      title: "Create a test",
      description: "Generate a tailored assessment with AI assistance.",
      cta: "Start building",
      icon: PlusCircle,
      path: "/create-test",
      cardClass:
        "border-primary-200 bg-primary-50/60 hover:border-primary-400 hover:bg-primary-50",
      iconClass: "bg-primary-500 text-white",
    },
    {
      eyebrow: "Practice",
      title: "Class 8 Science",
      description: "Work through adaptive MCQs from the question bank.",
      cta: "Start practicing",
      icon: Target,
      path: "/practice",
      cardClass:
        "border-secondary-200 bg-secondary-50/70 hover:border-secondary-400 hover:bg-secondary-50",
      iconClass: "bg-secondary-500 text-dark",
    },
    {
      eyebrow: "Speak",
      title: "AI Viva session",
      description: "Prepare with an interactive oral examination.",
      cta: "Enter viva",
      icon: Brain,
      path: "/viva",
      cardClass:
        "border-neutral-200 bg-white hover:border-neutral-400 hover:bg-neutral-50",
      iconClass: "bg-dark text-white",
    },
  ];

  return (
    <div className="min-h-screen bg-cream font-sans text-dark">
      <Header />

      {BYPASS_AUTH && (
        <div className="mx-auto max-w-7xl px-4 pt-4 sm:px-6 lg:px-8">
          <div
            className="flex flex-col gap-1 rounded-xl border border-yellow-300 bg-yellow-50 px-4 py-3 text-sm text-yellow-900 sm:flex-row sm:items-center sm:justify-between"
            role="alert"
          >
            <p className="font-semibold">Auth bypass enabled for local development</p>
            <p className="text-yellow-800">
              Remove VITE_BYPASS_AUTH before committing or deploying.
            </p>
          </div>
        </div>
      )}

      <main className="mx-auto max-w-7xl px-4 pb-16 pt-10 sm:px-6 lg:px-8 lg:pb-24 lg:pt-14">
        <section className="grid gap-8 lg:grid-cols-[minmax(0,1.15fr)_minmax(20rem,0.85fr)] lg:items-end">
          <div>
            <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-primary-200 bg-primary-50 px-3 py-1 text-xs font-bold uppercase tracking-[0.18em] text-primary-700">
              <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
              Learning studio
            </p>
            <h1 className="max-w-3xl text-4xl font-bold leading-[1.08] tracking-tight text-dark sm:text-5xl lg:text-6xl">
              Good to see you, <span className="text-primary-600">{firstName}</span>.
            </h1>
            <p className="mt-5 max-w-2xl text-base leading-7 text-neutral-600 sm:text-lg">
              Turn today&apos;s curiosity into a clear next step. Build a test,
              practice a topic, or sharpen your answers out loud.
            </p>
            <div className="mt-7 flex flex-col gap-3 sm:flex-row">
              <Button
                onClick={() => navigate("/create-test")}
                size="lg"
                className="w-full sm:w-auto"
              >
                <PlusCircle className="mr-2 h-5 w-5" aria-hidden="true" />
                Create a test
              </Button>
              <Button
                onClick={() => navigate("/practice")}
                size="lg"
                variant="outline"
                className="w-full sm:w-auto"
              >
                <Target className="mr-2 h-5 w-5" aria-hidden="true" />
                Explore practice
              </Button>
            </div>
          </div>

          <div className="relative isolate overflow-hidden rounded-3xl bg-dark p-5 text-white shadow-xl shadow-dark/10 sm:p-8">
            <div
              className="absolute -right-12 -top-16 -z-10 h-40 w-40 rounded-full bg-primary-500/30 blur-2xl"
              aria-hidden="true"
            />
            <div
              className="absolute -bottom-20 -left-12 -z-10 h-40 w-40 rounded-full bg-secondary-500/20 blur-2xl"
              aria-hidden="true"
            />
            <div className="flex items-center justify-between gap-4">
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary-300">
                Next best step
              </p>
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-white/10 text-secondary-300">
                <Sparkles className="h-5 w-5" aria-hidden="true" />
              </div>
            </div>
            <h2 className="mt-5 max-w-xs text-2xl font-bold leading-tight sm:mt-8 sm:text-3xl">
              Build momentum with a focused test.
            </h2>
            <p className="mt-3 max-w-sm text-sm leading-6 text-neutral-300 sm:mt-4">
              Start with a few questions and let Evater shape the session around
              your grade and goals.
            </p>
            <button
              type="button"
              onClick={() => navigate("/create-test")}
              className="mt-5 inline-flex min-h-11 items-center gap-2 rounded-xl bg-secondary-500 px-4 py-2.5 text-sm font-bold text-dark transition-colors hover:bg-secondary-400 focus:outline-none focus:ring-2 focus:ring-secondary-300 focus:ring-offset-2 focus:ring-offset-dark sm:mt-7"
            >
              Open test builder
              <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
            </button>
          </div>
        </section>

        <section
          aria-label="Account snapshot"
          className="mt-8 grid grid-cols-3 gap-2 sm:mt-10 sm:gap-3"
        >
          {accountStats.map((stat) => {
            const Icon = stat.icon;
            return (
              <div
                key={stat.label}
                className="flex min-h-[6.75rem] min-w-0 flex-col items-start gap-3 rounded-2xl border border-neutral-200 bg-white p-3 shadow-sm sm:min-h-[7.5rem] sm:flex-row sm:items-center sm:gap-4 sm:p-6"
              >
                <div
                  className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl sm:h-11 sm:w-11 ${stat.iconClass}`}
                >
                  <Icon className="h-5 w-5" aria-hidden="true" />
                </div>
                <div className="min-w-0">
                  <p className="text-[0.6rem] font-bold uppercase tracking-[0.12em] text-neutral-500 sm:text-xs sm:tracking-[0.16em]">
                    {stat.label}
                  </p>
                  <p
                    className="mt-1 truncate text-sm font-bold text-dark sm:text-xl"
                    title={String(stat.value)}
                  >
                    {stat.value}
                  </p>
                </div>
              </div>
            );
          })}
        </section>

        <section className="mt-16">
          <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary-700">
                Start here
              </p>
              <h2 className="mt-2 text-3xl font-bold tracking-tight text-dark">
                What are you working on?
              </h2>
            </div>
            <p className="text-sm font-medium text-neutral-500">Three ways to move forward</p>
          </div>

          <div className="grid gap-4 lg:grid-cols-3">
            {primaryActions.map((action) => {
              const Icon = action.icon;
              return (
                <button
                  key={action.path}
                  type="button"
                  onClick={() => navigate(action.path)}
                  className={`group flex min-h-[17rem] w-full flex-col rounded-3xl border p-6 text-left shadow-sm transition-all duration-200 hover:-translate-y-1 hover:shadow-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 sm:p-7 ${action.cardClass}`}
                >
                  <span
                    className={`flex h-12 w-12 items-center justify-center rounded-2xl ${action.iconClass}`}
                  >
                    <Icon className="h-6 w-6" aria-hidden="true" />
                  </span>
                  <span className="mt-8 flex items-start justify-between gap-4">
                    <span>
                      <span className="block text-xs font-bold uppercase tracking-[0.16em] text-neutral-500">
                        {action.eyebrow}
                      </span>
                      <span className="mt-2 block text-2xl font-bold leading-tight text-dark">
                        {action.title}
                      </span>
                    </span>
                    <ArrowUpRight
                      className="h-5 w-5 shrink-0 text-neutral-400 transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-dark"
                      aria-hidden="true"
                    />
                  </span>
                  <span className="mt-3 block text-sm leading-6 text-neutral-600">
                    {action.description}
                  </span>
                  <span className="mt-auto flex items-center gap-2 pt-6 text-sm font-bold text-dark">
                    {action.cta}
                    <ArrowRight
                      className="h-4 w-4 transition-transform group-hover:translate-x-1"
                      aria-hidden="true"
                    />
                  </span>
                </button>
              );
            })}
          </div>
        </section>

        <section className="mt-16 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(18rem,0.42fr)]">
          <div className="rounded-3xl border border-neutral-200 bg-white p-6 shadow-sm sm:p-8">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="text-xs font-bold uppercase tracking-[0.18em] text-primary-700">
                  Your workspace
                </p>
                <h2 className="mt-2 text-2xl font-bold tracking-tight text-dark sm:text-3xl">
                  Pick up where you left off
                </h2>
              </div>
              <button
                type="button"
                onClick={() => navigate("/previous-tests")}
                className="inline-flex min-h-11 items-center gap-1 self-start text-sm font-bold text-primary-700 transition-colors hover:text-primary-800 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 sm:self-auto"
              >
                View all tests
                <ArrowRight className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>

            <div className="mt-6 divide-y divide-neutral-100 border-y border-neutral-100">
              <div className="flex flex-col gap-4 py-5 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex min-w-0 items-start gap-4">
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-secondary-50 text-secondary-700">
                    <FileText className="h-5 w-5" aria-hidden="true" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-bold uppercase tracking-[0.16em] text-neutral-500">
                      Latest test
                    </p>
                    <p className="mt-1 truncate text-base font-bold text-dark">
                      {latestTestTitle}
                    </p>
                    <p className="mt-1 text-sm text-neutral-500">
                      {lastTest ? `Created ${formatDate(lastTest.created_at)}` : "Your generated tests will appear here."}
                    </p>
                  </div>
                </div>
                {lastTest ? (
                  <button
                    type="button"
                    onClick={() => navigate(`/view-test/${lastTest.id}`)}
                    className="inline-flex min-h-11 shrink-0 items-center gap-1 self-start rounded-lg px-3 text-sm font-bold text-primary-700 transition-colors hover:bg-primary-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 sm:self-auto"
                  >
                    Review
                    <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
                  </button>
                ) : (
                  <Button
                    type="button"
                    onClick={() => navigate("/create-test")}
                    size="sm"
                    variant="outline"
                    className="self-start sm:self-auto"
                  >
                    Create first test
                  </Button>
                )}
              </div>

              <div className="flex flex-col gap-4 py-5 sm:flex-row sm:items-center sm:justify-between">
                <div className="flex min-w-0 items-start gap-4">
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-primary-50 text-primary-700">
                    <MessageSquare className="h-5 w-5" aria-hidden="true" />
                  </div>
                  <div className="min-w-0">
                    <p className="text-xs font-bold uppercase tracking-[0.16em] text-neutral-500">
                      Latest feedback
                    </p>
                    <p className="mt-1 truncate text-base font-bold text-dark">
                      {latestFeedbackTitle}
                    </p>
                    <p className="mt-1 text-sm text-neutral-500">
                      {lastFeedback
                        ? "Open your latest evaluation to see the next step."
                        : "Evaluations and next steps will appear here."}
                    </p>
                  </div>
                </div>
                {lastFeedback ? (
                  <button
                    type="button"
                    onClick={() => navigate(`/view-feedback/${lastFeedback.id}`)}
                    className="inline-flex min-h-11 shrink-0 items-center gap-1 self-start rounded-lg px-3 text-sm font-bold text-primary-700 transition-colors hover:bg-primary-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 sm:self-auto"
                  >
                    Review
                    <ArrowUpRight className="h-4 w-4" aria-hidden="true" />
                  </button>
                ) : (
                  <Button
                    type="button"
                    onClick={() => navigate("/previous-feedbacks")}
                    size="sm"
                    variant="outline"
                    className="self-start sm:self-auto"
                  >
                    Browse feedback
                  </Button>
                )}
              </div>
            </div>
          </div>

          <aside className="rounded-3xl border border-primary-200 bg-primary-50 p-6 sm:p-8">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-white text-primary-600 shadow-sm">
              <Heart className="h-5 w-5" aria-hidden="true" />
            </div>
            <p className="mt-7 text-xs font-bold uppercase tracking-[0.18em] text-primary-700">
              Close the loop
            </p>
            <h2 className="mt-2 text-2xl font-bold leading-tight text-dark">
              Help shape the next version of Evater.
            </h2>
            <p className="mt-4 text-sm leading-6 text-neutral-600">
              Tell us what felt useful, what got in the way, or what you want to
              learn next.
            </p>
            <Button
              type="button"
              onClick={() => navigate("/general-feedback")}
              variant="secondary"
              className="mt-7 w-full"
            >
              <MessageSquare className="mr-2 h-4 w-4" aria-hidden="true" />
              Share feedback
            </Button>
          </aside>
        </section>
      </main>
    </div>
  );
}
