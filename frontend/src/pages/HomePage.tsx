import React from "react";
import { useNavigate } from "react-router-dom";
import {
  PlusCircle,
  FileText,
  MessageSquare,
  Heart,
  Sparkles,
  Clock,
  History,
  BarChart3,
  Coins,
  User,
  GraduationCap,
  School,
  Brain,
} from "lucide-react";
import { Card, CardContent } from "../components/ui/Card";
import { Header } from "../components/layout/Header";
// import { BlogSection } from '../components/blog/BlogSection'
import { useAppState } from "../contexts/AppStateContext";
import { useAuthContext } from "../contexts/AuthContext";
import { useProfile } from "../hooks/useProfile";
import { BYPASS_AUTH } from "../lib/auth/devBypass";

export function HomePage() {
  const navigate = useNavigate();
  const { appState } = useAppState();
  const { user } = useAuthContext();
  const { profile } = useProfile(user?.id);

  const navigationSections = [
    {
      title: "Give Tests",
      options: [
        {
          title: "Create Test",
          description: "Generate a new test with AI assistance",
          icon: PlusCircle,
          path: "/create-test",
          accentColor: "text-primary-600",
          bgColor: "bg-primary-50",
          borderColor: "group-hover:border-primary-200",
        },
        {
          title: "AI Viva Session",
          description: "Interactive oral examination with AI",
          icon: Brain,
          path: "/viva",
          accentColor: "text-purple-600",
          bgColor: "bg-purple-50",
          borderColor: "group-hover:border-purple-200",
        },
      ],
    },
    {
      title: "Previous Tests",
      options: [
        {
          title: "View Last Generated Test",
          description: appState.last_generated_test
            ? `${appState.last_generated_test.subject} - ${appState.last_generated_test.chapter}`
            : "No test generated yet",
          icon: FileText,
          path: appState.last_generated_test
            ? `/view-test/${appState.last_generated_test.id}`
            : "/view-test",
          accentColor: "text-secondary-600",
          bgColor: "bg-secondary-50",
          borderColor: "group-hover:border-secondary-200",
          disabled: !appState.last_generated_test,
        },
        {
          title: "Previous Tests",
          description: "View all your previously created tests",
          icon: History,
          path: "/previous-tests",
          accentColor: "text-purple-600",
          bgColor: "bg-purple-50",
          borderColor: "group-hover:border-purple-200",
        },
      ],
    },
    {
      title: "Previous Feedbacks",
      options: [
        {
          title: "View Last Feedback",
          description: appState.last_generated_feedback
            ? `Review evaluation from ${new Date(
                appState.last_generated_feedback.created_at
              ).toLocaleDateString()}`
            : "No feedback available yet",
          icon: MessageSquare,
          path: appState.last_generated_feedback
            ? `/view-feedback/${appState.last_generated_feedback.id}`
            : "/view-feedback",
          accentColor: "text-neutral-600",
          bgColor: "bg-neutral-100",
          borderColor: "group-hover:border-neutral-200",
          disabled: !appState.last_generated_feedback,
        },
        {
          title: "Previous Feedbacks",
          description: "View all your previous test evaluations",
          icon: BarChart3,
          path: "/previous-feedbacks",
          accentColor: "text-blue-600",
          bgColor: "bg-blue-50",
          borderColor: "group-hover:border-blue-200",
        },
      ],
    },
  ];

  const handleNavigation = (path: string, disabled?: boolean) => {
    if (!disabled) {
      navigate(path);
    }
  };

  return (
    <div className="min-h-screen bg-cream font-sans">
      <Header />
      {BYPASS_AUTH && (
        <div
          className="mx-4 sm:mx-6 lg:mx-8 mt-6 mb-2 rounded-lg border border-yellow-300 bg-yellow-50 px-4 py-3 text-sm text-yellow-800"
          role="alert"
        >
          <p className="font-semibold">
            Auth bypass enabled for local development
          </p>
          <p>
            Remove VITE_BYPASS_AUTH before committing or deploying to
            production.
          </p>
        </div>
      )}

      <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        {/* Welcome Section */}
        <div className="text-center mb-16">
          <div className="inline-flex items-center justify-center mb-6">
            <img
              src="/Evater_logo_2.png"
              alt="Evater Logo"
              className="w-56 h-auto object-contain"
            />
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-dark mb-6 tracking-tight">
            Welcome back,{" "}
            <span className="text-primary-600">
              {profile?.user_name
                ? profile.user_name.split(" ")[0]
                : "Educator"}
            </span>
          </h1>
          <p className="text-xl text-neutral-500 max-w-2xl mx-auto leading-relaxed font-medium">
            Ready to create, evaluate, and enhance your learning experience
          </p>
        </div>

        {/* Account Status Section */}
        <div className="mb-16">
          <div className="bg-white rounded-2xl shadow-sm border border-neutral-100 p-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 divide-y md:divide-y-0 md:divide-x divide-neutral-100">
              {/* Grade */}
              <div className="flex flex-col items-center justify-center p-4">
                <div className="flex items-center mb-3">
                  <div className="p-2 bg-neutral-50 rounded-lg mr-3">
                    <GraduationCap className="h-5 w-5 text-neutral-600" />
                  </div>
                  <span className="text-sm font-semibold text-neutral-500 uppercase tracking-wider">
                    Grade
                  </span>
                </div>
                <div className="text-2xl font-bold text-dark">
                  {profile?.grade || "N/A"}
                </div>
              </div>

              {/* School */}
              <div className="flex flex-col items-center justify-center p-4">
                <div className="flex items-center mb-3">
                  <div className="p-2 bg-neutral-50 rounded-lg mr-3">
                    <School className="h-5 w-5 text-neutral-600" />
                  </div>
                  <span className="text-sm font-semibold text-neutral-500 uppercase tracking-wider">
                    School
                  </span>
                </div>
                <div
                  className="text-2xl font-bold text-dark text-center truncate max-w-full px-4"
                  title={profile?.school || "Not set"}
                >
                  {profile?.school || "Not set"}
                </div>
              </div>

              {/* Credits */}
              <div className="flex flex-col items-center justify-center p-4">
                <div className="flex items-center mb-3">
                  <div className="p-2 bg-yellow-50 rounded-lg mr-3">
                    <Coins className="h-5 w-5 text-yellow-600" />
                  </div>
                  <span className="text-sm font-semibold text-neutral-500 uppercase tracking-wider">
                    Credits
                  </span>
                </div>
                <div className="flex flex-col items-center">
                  <span className="text-2xl font-bold text-dark">
                    {profile?.credits ?? "---"}
                  </span>
                  {profile?.credits !== null &&
                    profile?.credits !== undefined &&
                    profile.credits <= 5 && (
                      <span className="mt-2 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                        {profile.credits === 0 ? "No Credits" : "Low Credits"}
                      </span>
                    )}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="space-y-16 mb-16">
          {navigationSections.map((section) => (
            <div key={section.title}>
              <div className="flex items-center mb-8">
                <h2 className="text-2xl font-bold text-dark">
                  {section.title}
                </h2>
                <div className="ml-4 h-px bg-neutral-200 flex-grow"></div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                {section.options.map((option) => {
                  const Icon = option.icon;
                  const isDisabled = option.disabled;
                  return (
                    <div
                      key={option.path}
                      className={`group relative bg-white rounded-2xl p-6 border border-neutral-100 shadow-sm transition-all duration-300 
                        ${
                          isDisabled
                            ? "opacity-60 cursor-not-allowed bg-neutral-50"
                            : "hover:shadow-md hover:-translate-y-1 cursor-pointer " +
                              option.borderColor
                        }`}
                      onClick={() => handleNavigation(option.path, isDisabled)}
                    >
                      <div className="flex items-start space-x-4">
                        <div
                          className={`flex-shrink-0 w-12 h-12 rounded-xl flex items-center justify-center transition-colors duration-300 ${
                            isDisabled ? "bg-neutral-200" : option.bgColor
                          }`}
                        >
                          <Icon
                            className={`h-6 w-6 ${
                              isDisabled
                                ? "text-neutral-400"
                                : option.accentColor
                            }`}
                          />
                        </div>
                        <div className="flex-1 min-w-0">
                          <h3
                            className={`text-lg font-bold text-dark mb-1 ${
                              !isDisabled &&
                              "group-hover:text-primary-600 transition-colors"
                            }`}
                          >
                            {option.title}
                          </h3>
                          <p className="text-sm text-neutral-500 leading-relaxed">
                            {option.description}
                          </p>
                          {isDisabled && (
                            <span className="inline-block mt-2 bg-neutral-200 text-neutral-600 px-2 py-0.5 rounded text-xs font-medium">
                              Not Available
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* General Feedback Section */}
        <div className="bg-white rounded-2xl shadow-sm border border-neutral-100 p-10 text-center max-w-3xl mx-auto">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-50 mb-6">
            <Heart className="h-8 w-8 text-primary-500" />
          </div>
          <h2 className="text-2xl font-bold text-dark mb-4">Help Us Improve</h2>
          <p className="text-neutral-500 mb-8 max-w-xl mx-auto leading-relaxed">
            Your feedback helps us make Evater better for everyone. Share your
            thoughts, suggestions, or report any issues to help us create the
            perfect learning experience.
          </p>
          <button
            onClick={() => navigate("/general-feedback")}
            className="inline-flex items-center px-8 py-3 bg-dark text-white font-semibold rounded-xl hover:bg-neutral-800 transition-all duration-200 shadow-lg shadow-neutral-200 hover:shadow-xl transform hover:-translate-y-0.5"
          >
            <MessageSquare className="h-5 w-5 mr-2.5" />
            Share Your Feedback
          </button>
        </div>
      </div>
    </div>
  );
}
