import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../components/ui/Button";
import { Select } from "../components/ui/Select";
import { Card, CardContent } from "../components/ui/Card";
import { Header } from "../components/layout/Header";
import { useAuthContext } from "../contexts/AuthContext";
import { useAppState } from "../contexts/AppStateContext";
import { supabase } from "../lib/supabase";
import { TestGenerationParams, Question, ChapterContent } from "../types";
import {
  BookOpen,
  Calculator,
  FlaskConical,
  History as HistoryIcon,
  Globe,
  Languages,
  Brain,
  Atom,
  Dna,
  Zap,
  CheckCircle2,
  Clock,
  AlertCircle,
  ChevronRight,
  GraduationCap,
  Layout,
  FileText,
  Sparkles,
} from "lucide-react";

interface ApiQuestion {
  question_text: string;
  question_type:
    | "short_answer"
    | "long_answer"
    | "mcq_single"
    | "mcq_multi"
    | "true_false";
  question_number: number;
  maximum_marks: number;
  difficulty: "Easy" | "Medium" | "Hard";
  contains_math_expression: boolean;
  options?: string[] | null;
}

interface ApiResponse {
  questions: ApiQuestion[];
}

export function CreateTestPage() {
  const navigate = useNavigate();
  const { user } = useAuthContext();
  const { setLastGeneratedTest } = useAppState();

  const [selectedGrade, setSelectedGrade] = useState("");
  const [subject, setSubject] = useState("");
  const [chapter, setChapter] = useState("");
  const [difficulty, setDifficulty] = useState<"Easy" | "Medium" | "Hard">(
    "Medium"
  );
  const [length, setLength] = useState<"Short" | "Long">("Short");
  const [testType, setTestType] = useState<
    "objective" | "subjective" | "mixed"
  >("mixed");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [chapterContents, setChapterContents] = useState<ChapterContent[]>([]);
  const [availableSubjects, setAvailableSubjects] = useState<string[]>([]);
  const [availableChapters, setAvailableChapters] = useState<string[]>([]);

  const gradeOptions = Array.from({ length: 12 }, (_, i) => (i + 1).toString());

  const testTypeOptions = [
    {
      id: "objective",
      label: "Objective (MCQ)",
      icon: CheckCircle2,
      description: "Multiple choice questions only",
    },
    {
      id: "subjective",
      label: "Subjective (Theory)",
      icon: FileText,
      description: "Long and short answer questions",
    },
    {
      id: "mixed",
      label: "Mixed (Standard)",
      icon: Brain,
      description: "Combination of both types",
    },
  ];

  useEffect(() => {
    fetchChapterContents();
  }, []);

  useEffect(() => {
    if (selectedGrade) {
      const subjects = chapterContents
        .filter((content) => content.grade === selectedGrade)
        .map((content) => content.subject)
        .filter(
          (subject, index, self) => subject && self.indexOf(subject) === index
        );
      setAvailableSubjects(subjects as string[]);
      setSubject("");
      setChapter("");
    }
  }, [selectedGrade, chapterContents]);

  useEffect(() => {
    if (selectedGrade && subject) {
      const chapters = chapterContents
        .filter(
          (content) =>
            content.grade === selectedGrade && content.subject === subject
        )
        .map((content) => content.chapter)
        .filter(
          (chapter, index, self) => chapter && self.indexOf(chapter) === index
        );
      setAvailableChapters(chapters as string[]);
      setChapter("");
    }
  }, [selectedGrade, subject, chapterContents]);

  const fetchChapterContents = async () => {
    try {
      const { data, error } = await supabase
        .from("Chapter_contents")
        .select("*");

      if (error) throw error;
      setChapterContents(data || []);
    } catch (err) {
      console.error("Error fetching chapter contents:", err);
    }
  };

  const getSubjectIcon = (subjectName: string) => {
    const s = subjectName.toLowerCase();
    if (s.includes("math")) return Calculator;
    if (s.includes("science")) return FlaskConical;
    if (s.includes("history")) return HistoryIcon;
    if (s.includes("geography")) return Globe;
    if (s.includes("english")) return Languages;
    if (s.includes("physics")) return Atom;
    if (s.includes("chemistry")) return FlaskConical;
    if (s.includes("biology")) return Dna;
    if (s.includes("computer") || s.includes("tech")) return Layout;
    return BookOpen;
  };

  const callGenerateQuestionsAPI = async (params: {
    grade: string;
    subject: string;
    topic: string;
    difficulty_level: string;
    length: string;
    test_type: string;
    special_instructions: string[];
  }): Promise<ApiQuestion[]> => {
    const apiUrl =
      process.env.NODE_ENV === "production"
        ? "/.netlify/functions/generate-questions"
        : "/api/external/api/gen_question";

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(params),
    });

    if (!response.ok) {
      if (response.status === 400) {
        throw new Error(
          "Missing required fields. Please fill in all form fields."
        );
      } else if (response.status === 500) {
        throw new Error("Server error. Please try again later.");
      } else {
        throw new Error(`API error: ${response.status}`);
      }
    }

    const data: ApiResponse = await response.json();
    return data.questions;
  };

  const handleSubmit = async () => {
    setError("");
    setLoading(true);

    try {
      if (!selectedGrade || !subject || !chapter) {
        throw new Error("Please fill in all required fields");
      }

      const apiPayload = {
        grade: selectedGrade,
        subject: subject,
        topic: chapter,
        difficulty_level: difficulty,
        length: length,
        test_type: testType,
        special_instructions: [], // Kept empty as we moved to explicit test types
      };

      const questions = await callGenerateQuestionsAPI(apiPayload);

      const convertedQuestions: Question[] = questions.map((q) => ({
        question_text: q.question_text,
        question_type: q.question_type,
        question_number: q.question_number,
        maximum_marks: q.maximum_marks,
        difficulty: q.difficulty.toLowerCase() as "easy" | "medium" | "hard",
        contains_math_expression: q.contains_math_expression,
        options: q.options || undefined,
      }));

      const { data, error } = await supabase
        .from("Questions_Created")
        .insert([
          {
            created_by: user!.id,
            grade: selectedGrade,
            subject: subject,
            chapter: chapter,
            difficulty_level: difficulty.toLowerCase(),
            length: length.toLowerCase(),
            special_instructions: [testType], // Storing test type in special instructions for backward compatibility/reference
            test: convertedQuestions,
          },
        ])
        .select()
        .single();

      if (error) throw error;

      setLastGeneratedTest(data);

      // Route based on test type
      if (testType === "objective") {
        navigate(`/take-test/${data.id}`);
      } else {
        navigate(`/view-test/${data.id}`);
      }
    } catch (err) {
      console.error("Error generating test:", err);
      setError(
        err instanceof Error
          ? err.message
          : "Failed to generate test. Please try again."
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-cream font-sans pb-20">
      <Header />

      <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-primary-50 mb-4">
            <Sparkles className="h-8 w-8 text-primary-600" />
          </div>
          <h1 className="text-3xl font-bold text-dark mb-2">Create New Test</h1>
          <p className="text-neutral-500 max-w-lg mx-auto">
            Customize your assessment by selecting the grade, subject, and
            difficulty.
          </p>
        </div>

        <div className="space-y-12">
          {/* Grade Selection */}
          <section>
            <h2 className="text-lg font-semibold text-dark mb-4 flex items-center">
              <GraduationCap className="w-5 h-5 mr-2 text-primary-600" />
              Select Grade
            </h2>
            <div className="grid grid-cols-6 md:grid-cols-12 gap-3">
              {gradeOptions.map((grade) => (
                <button
                  key={grade}
                  onClick={() => setSelectedGrade(grade)}
                  className={`
                    aspect-square rounded-xl flex items-center justify-center text-lg font-bold transition-all duration-200
                    ${
                      selectedGrade === grade
                        ? "bg-primary-600 text-white shadow-lg scale-105"
                        : "bg-white text-neutral-600 hover:bg-primary-50 hover:text-primary-600 border border-neutral-200"
                    }
                  `}
                >
                  {grade}
                </button>
              ))}
            </div>
          </section>

          {/* Subject Selection */}
          <section
            className={`transition-opacity duration-300 ${
              !selectedGrade ? "opacity-50 pointer-events-none" : ""
            }`}
          >
            <h2 className="text-lg font-semibold text-dark mb-4 flex items-center">
              <BookOpen className="w-5 h-5 mr-2 text-primary-600" />
              Select Subject
            </h2>
            {availableSubjects.length > 0 ? (
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                {availableSubjects.map((s) => {
                  const Icon = getSubjectIcon(s);
                  return (
                    <button
                      key={s}
                      onClick={() => setSubject(s)}
                      className={`
                        p-4 rounded-xl border-2 text-left transition-all duration-200 flex flex-col items-start gap-3
                        ${
                          subject === s
                            ? "border-primary-600 bg-primary-50 shadow-md"
                            : "border-transparent bg-white hover:border-primary-100 hover:shadow-sm"
                        }
                      `}
                    >
                      <div
                        className={`p-2 rounded-lg ${
                          subject === s
                            ? "bg-white text-primary-600"
                            : "bg-primary-50 text-primary-600"
                        }`}
                      >
                        <Icon className="w-6 h-6" />
                      </div>
                      <span
                        className={`font-semibold ${
                          subject === s
                            ? "text-primary-900"
                            : "text-neutral-700"
                        }`}
                      >
                        {s}
                      </span>
                    </button>
                  );
                })}
              </div>
            ) : (
              <div className="bg-white rounded-xl p-8 text-center border border-dashed border-neutral-300">
                <p className="text-neutral-400">
                  Select a grade to view subjects
                </p>
              </div>
            )}
          </section>

          {/* Chapter Selection */}
          <section
            className={`transition-opacity duration-300 ${
              !subject ? "opacity-50 pointer-events-none" : ""
            }`}
          >
            <h2 className="text-lg font-semibold text-dark mb-4 flex items-center">
              <Layout className="w-5 h-5 mr-2 text-primary-600" />
              Select Topic/Chapter
            </h2>
            <div className="bg-white p-1 rounded-xl shadow-sm border border-neutral-200">
              <Select
                value={chapter}
                onChange={(e) => setChapter(e.target.value)}
                options={[
                  { value: "", label: "Select topic/chapter" },
                  ...availableChapters.map((c) => ({ value: c, label: c })),
                ]}
                className="border-0 focus:ring-0 text-lg py-3"
                disabled={!subject}
              />
            </div>
          </section>

          {/* Difficulty & Length */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <section>
              <h2 className="text-lg font-semibold text-dark mb-4 flex items-center">
                <Brain className="w-5 h-5 mr-2 text-primary-600" />
                Difficulty
              </h2>
              <div className="grid grid-cols-3 gap-3">
                {["Easy", "Medium", "Hard"].map((level) => (
                  <button
                    key={level}
                    onClick={() => setDifficulty(level as any)}
                    className={`
                      py-3 px-2 rounded-xl text-sm font-bold transition-all duration-200 border-2
                      ${
                        difficulty === level
                          ? level === "Easy"
                            ? "border-green-500 bg-green-50 text-green-700"
                            : level === "Medium"
                            ? "border-yellow-500 bg-yellow-50 text-yellow-700"
                            : "border-red-500 bg-red-50 text-red-700"
                          : "border-transparent bg-white text-neutral-500 hover:bg-neutral-50"
                      }
                    `}
                  >
                    {level}
                  </button>
                ))}
              </div>
            </section>

            <section>
              <h2 className="text-lg font-semibold text-dark mb-4 flex items-center">
                <Clock className="w-5 h-5 mr-2 text-primary-600" />
                Length
              </h2>
              <div className="grid grid-cols-2 gap-3">
                {["Short", "Long"].map((l) => (
                  <button
                    key={l}
                    onClick={() => setLength(l as any)}
                    className={`
                      py-3 px-4 rounded-xl text-sm font-bold transition-all duration-200 border-2
                      ${
                        length === l
                          ? "border-primary-600 bg-primary-50 text-primary-700"
                          : "border-transparent bg-white text-neutral-500 hover:bg-neutral-50"
                      }
                    `}
                  >
                    {l}{" "}
                    <span className="block text-xs font-normal opacity-70 mt-1">
                      {l === "Short" ? "~5 Qs" : "~10 Qs"}
                    </span>
                  </button>
                ))}
              </div>
            </section>
          </div>

          {/* Test Type Selection */}
          <section>
            <h2 className="text-lg font-semibold text-dark mb-4 flex items-center">
              <Brain className="w-5 h-5 mr-2 text-primary-600" />
              Test Type
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {testTypeOptions.map((opt) => {
                const Icon = opt.icon;
                const isSelected = testType === opt.id;
                return (
                  <button
                    key={opt.id}
                    onClick={() => setTestType(opt.id as any)}
                    className={`
                      flex flex-col items-start p-4 rounded-xl border-2 transition-all duration-200 text-left
                      ${
                        isSelected
                          ? "bg-primary-600 text-white border-primary-600 shadow-md"
                          : "bg-white text-neutral-600 border-neutral-200 hover:border-primary-200 hover:bg-neutral-50"
                      }
                    `}
                  >
                    <div className="flex items-center mb-2">
                      <Icon
                        className={`w-5 h-5 mr-2 ${
                          isSelected ? "text-white" : "text-primary-600"
                        }`}
                      />
                      <span className="font-bold">{opt.label}</span>
                    </div>
                    <p
                      className={`text-xs ${
                        isSelected ? "text-primary-100" : "text-neutral-500"
                      }`}
                    >
                      {opt.description}
                    </p>
                  </button>
                );
              })}
            </div>
          </section>

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start">
              <AlertCircle className="w-5 h-5 text-red-600 mr-3 mt-0.5" />
              <p className="text-sm text-red-600">{error}</p>
            </div>
          )}

          <div className="flex items-center justify-end pt-6 border-t border-neutral-200">
            <Button
              variant="outline"
              onClick={() => navigate("/home")}
              className="mr-4 border-neutral-300 text-neutral-600 hover:bg-neutral-50"
            >
              Cancel
            </Button>
            <Button
              onClick={handleSubmit}
              loading={loading}
              disabled={!selectedGrade || !subject || !chapter}
              className="px-8 py-6 text-lg rounded-xl shadow-lg shadow-primary-200 hover:shadow-xl hover:-translate-y-0.5 transition-all"
            >
              {loading ? "Generating Test..." : "Generate Test"}
              {!loading && <ChevronRight className="ml-2 w-5 h-5" />}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
