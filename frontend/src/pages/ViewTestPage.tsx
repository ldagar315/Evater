import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  FileCheck,
  ArrowLeft,
  Upload,
  PenTool,
  ChevronDown,
  ChevronUp,
  FileText,
} from "lucide-react";
import { Button } from "../components/ui/Button";
import { Card, CardContent, CardHeader } from "../components/ui/Card";
import { Header } from "../components/layout/Header";
import { useAuthContext } from "../contexts/AuthContext";
import { useAppState } from "../contexts/AppStateContext";
import { supabase } from "../lib/supabase";
import { QuestionsCreated, FeedbackResponse } from "../types";
import { QuestionCard } from "../components/test/QuestionCard";

interface ApiFeedbackResponse {
  merged: FeedbackResponse[];
}

export function ViewTestPage() {
  const { testId } = useParams();
  const navigate = useNavigate();
  const { user } = useAuthContext();
  const { setLastGeneratedFeedback } = useAppState();

  const [test, setTest] = useState<QuestionsCreated | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Mode selection
  const [mode, setMode] = useState<"interactive" | "upload" | null>(null);
  const [showInstructions, setShowInstructions] = useState(true);

  // Interactive Mode State
  const [answers, setAnswers] = useState<Record<number, any>>({});
  const [submittingInteractive, setSubmittingInteractive] = useState(false);

  // Upload Mode State
  const [files, setFiles] = useState<File[]>([]);
  const [uploadingToBucket, setUploadingToBucket] = useState(false);
  const [uploadedFileUrls, setUploadedFileUrls] = useState<string[]>([]);
  const [evaluatingUpload, setEvaluatingUpload] = useState(false);

  useEffect(() => {
    if (testId) {
      fetchTest();
    } else {
      fetchLatestTest();
    }
  }, [testId, user]);

  const fetchTest = async () => {
    if (!testId) return;
    try {
      setLoading(true);
      setError("");
      const { data, error } = await supabase
        .from("Questions_Created")
        .select("*")
        .eq("id", parseInt(testId))
        .eq("created_by", user!.id)
        .single();

      if (error) throw error;
      setTest(data);
    } catch (err) {
      console.error("Error fetching test:", err);
      setError("Test not found or you do not have permission to view it");
    } finally {
      setLoading(false);
    }
  };

  const fetchLatestTest = async () => {
    try {
      setLoading(true);
      setError("");
      const { data, error } = await supabase
        .from("Questions_Created")
        .select("*")
        .eq("created_by", user!.id)
        .order("created_at", { ascending: false })
        .limit(1)
        .single();

      if (error) throw error;
      setTest(data);
    } catch (err) {
      setError("No tests found. Create your first test to get started!");
    } finally {
      setLoading(false);
    }
  };

  // --- Interactive Mode Handlers ---

  const handleAnswerChange = (questionNumber: number, value: any) => {
    setAnswers((prev) => ({
      ...prev,
      [questionNumber]: value,
    }));
  };

  const submitInteractiveTest = async () => {
    if (!test) return;
    setSubmittingInteractive(true);
    setError("");

    try {
      const questions = Array.isArray(test.test) ? test.test : [];

      // 1. Upload any image answers first and get URLs
      const uploadedImages: Record<number, string> = {};

      for (const q of questions) {
        const ans = answers[q.question_number];
        if (ans?.image?.blob) {
          const fileName = `${user!.id}/${test.id}/q${
            q.question_number
          }_${Date.now()}.jpg`;
          const { error: uploadError } = await supabase.storage
            .from("answer-sheets")
            .upload(fileName, ans.image.blob);

          if (uploadError) throw uploadError;

          const { data: urlData } = supabase.storage
            .from("answer-sheets")
            .getPublicUrl(fileName);

          uploadedImages[q.question_number] = urlData.publicUrl;
        }
      }

      // 2. Construct payload for direct feedback generation
      const feedbackPayload = {
        questions: questions.map((q: any) => ({
          question_text: q.question_text,
          question_type: q.question_type,
          question_number: q.question_number,
          maximum_marks: q.maximum_marks,
          difficulty: capitalizeDifficulty(q.difficulty),
          contains_math_expression: q.contains_math_expression,
          options: q.options || null,
        })),
        answers: questions.map((q: any) => {
          const ans = answers[q.question_number];
          return {
            question_number: q.question_number,
            answer_text: ans?.text || "",
            image_url: uploadedImages[q.question_number] || null,
          };
        }),
      };

      // 3. Call direct feedback API
      const apiUrl =
        process.env.NODE_ENV === "production"
          ? "/.netlify/functions/generate-feedback-direct"
          : "/api/external/api/gen_feedback_direct";

      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(feedbackPayload),
      });

      if (!response.ok) throw new Error("Failed to generate feedback");

      const data = await response.json();

      // 4. Save feedback to Supabase
      const { data: feedbackData, error: feedbackError } = await supabase
        .from("FeedbackTest")
        .insert([
          {
            created_by: user!.id,
            test_id: parseInt(testId!),
            feedback_json: data.feedback,
            total_marks: questions.reduce(
              (acc: number, q: any) => acc + q.maximum_marks,
              0
            ),
            marks_obtained: 0, // Placeholder, ideally backend should return this
          },
        ])
        .select()
        .single();

      if (feedbackError) throw feedbackError;

      // 5. Navigate to feedback view
      navigate(`/view-feedback/${feedbackData.id}`);
    } catch (err) {
      console.error("Submission error:", err);
      setError("Failed to submit test. Please try again.");
    } finally {
      setSubmittingInteractive(false);
    }
  };

  // --- Upload Mode Handlers ---

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files));
    }
  };

  const uploadToBucket = async () => {
    if (files.length === 0) return;
    setUploadingToBucket(true);
    try {
      const bucketName = "question-papers-test";
      const folderName = "question-test-v1";
      const urls: string[] = [];

      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const fileName = `${folderName}/${Date.now()}_${i}_${file.name}`;

        const { error } = await supabase.storage
          .from(bucketName)
          .upload(fileName, file);

        if (error) throw error;

        const { data } = supabase.storage
          .from(bucketName)
          .getPublicUrl(fileName);

        if (data) urls.push(data.publicUrl);
      }
      setUploadedFileUrls(urls);
    } catch (err) {
      console.error("Upload error:", err);
      setError("Failed to upload files");
    } finally {
      setUploadingToBucket(false);
    }
  };

  const submitUploadTest = async () => {
    if (!test) return;
    setEvaluatingUpload(true);
    try {
      // If we have bucket URLs, use them. Else upload to 'answer-sheets' bucket (per original logic)
      let finalUrls = [...uploadedFileUrls];

      if (finalUrls.length === 0 && files.length > 0) {
        const uploadPromises = files.map(async (file, index) => {
          const fileName = `${user!.id}/${
            test.id
          }/sheet_${index}_${Date.now()}.${file.name.split(".").pop()}`;
          const { error } = await supabase.storage
            .from("answer-sheets")
            .upload(fileName, file);
          if (error) throw error;
          const { data } = supabase.storage
            .from("answer-sheets")
            .getPublicUrl(fileName);
          return data.publicUrl;
        });
        finalUrls = await Promise.all(uploadPromises);
      }

      if (finalUrls.length === 0) {
        setError("No files to evaluate");
        return;
      }

      const questions = Array.isArray(test.test) ? test.test : [];
      const formattedQuestions = questions.map((q: any) => ({
        question_text: q.question_text,
        question_type: q.question_type,
        question_number: q.question_number,
        maximum_marks: q.maximum_marks,
        difficulty: capitalizeDifficulty(q.difficulty),
        contains_math_expression: q.contains_math_expression,
        options: q.options || null,
      }));

      await processSubmission(finalUrls, formattedQuestions);
    } catch (err) {
      console.error("Evaluation error:", err);
      setError("Failed to evaluate. Please try again.");
    } finally {
      setEvaluatingUpload(false);
    }
  };

  // --- Common Submission Logic ---

  const capitalizeDifficulty = (d: string) => {
    const map: Record<string, string> = {
      easy: "Easy",
      medium: "Medium",
      hard: "Hard",
    };
    return map[d.toLowerCase()] || "Medium";
  };

  const processSubmission = async (imageUrls: string[], questions: any[]) => {
    // API Call
    const payload = {
      image_url: imageUrls,
      questions: { questions },
    };

    const apiUrl =
      process.env.NODE_ENV === "production"
        ? "/.netlify/functions/generate-feedback"
        : "/api/external/api/gen_answer";

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) throw new Error(`API Error: ${response.status}`);

    const data: ApiFeedbackResponse = await response.json();

    // Save to DB
    const { data: feedbackData, error: dbError } = await supabase
      .from("FeedbackTest")
      .insert([
        {
          given_by: user!.id,
          for_test: test!.id,
          feedback: data.merged,
        },
      ])
      .select()
      .single();

    if (dbError) throw dbError;

    setLastGeneratedFeedback(feedbackData);
    navigate(`/view-feedback/${feedbackData.id}`);
  };

  if (loading) return <div className="p-8 text-center">Loading...</div>;
  if (error || !test)
    return (
      <div className="p-8 text-center text-red-600">
        {error || "Test not found"}
      </div>
    );

  const questions = Array.isArray(test.test) ? test.test : [];

  return (
    <div className="min-h-screen bg-cream">
      <Header />
      <div className="max-w-6xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        {/* Header Section */}
        <div className="mb-8">
          <Button
            variant="ghost"
            onClick={() => navigate("/previous-tests")}
            className="mb-4"
          >
            <ArrowLeft className="h-4 w-4 mr-2" /> Back
          </Button>
          <div className="flex justify-between items-end">
            <div>
              <h1 className="text-3xl font-bold text-dark mb-2">
                {test.chapter}
              </h1>
              <div className="flex gap-3 text-sm text-neutral-600">
                <span className="bg-white px-2 py-1 rounded border">
                  {test.subject}
                </span>
                <span className="bg-white px-2 py-1 rounded border">
                  Class {test.grade}
                </span>
                <span className="bg-white px-2 py-1 rounded border capitalize">
                  {test.difficulty_level}
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold text-primary-600">
                {questions.length}
              </div>
              <div className="text-xs text-neutral-500 uppercase tracking-wide">
                Questions
              </div>
            </div>
          </div>
        </div>

        {/* Instructions Collapsible */}
        <Card className="mb-8 border-l-4 border-l-primary-500">
          <CardHeader
            className="cursor-pointer flex flex-row items-center justify-between p-4"
            onClick={() => setShowInstructions(!showInstructions)}
          >
            <h3 className="font-semibold text-lg flex items-center">
              <FileCheck className="w-5 h-5 mr-2 text-primary-600" />
              Test Instructions & Modes
            </h3>
            {showInstructions ? (
              <ChevronUp className="w-5 h-5" />
            ) : (
              <ChevronDown className="w-5 h-5" />
            )}
          </CardHeader>
          {showInstructions && (
            <CardContent className="pt-0 pb-4 px-4 text-neutral-600">
              <p className="mb-2">You can attempt this test in two ways:</p>
              <ul className="list-disc list-inside space-y-1 ml-2">
                <li>
                  <strong>Interactive Mode:</strong> Attempt questions one by
                  one directly on the screen. You can type answers, select MCQs,
                  use voice typing, or upload images for specific questions.
                </li>
                <li>
                  <strong>Upload Mode:</strong> Write your answers on physical
                  paper, scan/click photos of the pages, and upload them in bulk
                  for evaluation.
                </li>
              </ul>
              {test.special_instructions && (
                <div className="mt-4 pt-4 border-t">
                  <span className="font-medium text-dark">
                    Special Instructions:{" "}
                  </span>
                  {test.special_instructions.join(", ")}
                </div>
              )}
            </CardContent>
          )}
        </Card>

        {/* Mode Selection */}
        {!mode && (
          <div className="grid md:grid-cols-2 gap-6">
            <div
              onClick={() => setMode("interactive")}
              className="bg-white p-8 rounded-xl shadow-sm border-2 border-transparent hover:border-primary-500 cursor-pointer transition-all hover:shadow-md group text-center"
            >
              <div className="w-20 h-20 bg-primary-50 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
                <PenTool className="w-10 h-10 text-primary-600" />
              </div>
              <h3 className="text-xl font-bold text-dark mb-2">
                Interactive Mode
              </h3>
              <p className="text-neutral-500">
                Attempt questions digitally with smart inputs (Text, Voice,
                Image)
              </p>
            </div>

            <div
              onClick={() => setMode("upload")}
              className="bg-white p-8 rounded-xl shadow-sm border-2 border-transparent hover:border-green-500 cursor-pointer transition-all hover:shadow-md group text-center"
            >
              <div className="w-20 h-20 bg-green-50 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform">
                <Upload className="w-10 h-10 text-green-600" />
              </div>
              <h3 className="text-xl font-bold text-dark mb-2">
                Upload Answer Sheets
              </h3>
              <p className="text-neutral-500">
                Bulk upload scanned pages or photos of your handwritten answers
              </p>
            </div>
          </div>
        )}

        {/* Interactive Mode UI */}
        {mode === "interactive" && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold">Interactive Test</h2>
              <Button variant="ghost" size="sm" onClick={() => setMode(null)}>
                Change Mode
              </Button>
            </div>

            <div className="space-y-6">
              {questions.map((q: any) => (
                <QuestionCard
                  key={q.question_number}
                  question={q}
                  answer={answers[q.question_number]}
                  onAnswerChange={(val) =>
                    handleAnswerChange(q.question_number, val)
                  }
                />
              ))}
            </div>

            <div className="mt-8 flex justify-end">
              <Button
                size="lg"
                onClick={submitInteractiveTest}
                loading={submittingInteractive}
                className="w-full md:w-auto"
              >
                Submit Test for Evaluation
              </Button>
            </div>
          </div>
        )}

        {/* Upload Mode UI */}
        {mode === "upload" && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-bold">Upload Answer Sheets</h2>
              <Button variant="ghost" size="sm" onClick={() => setMode(null)}>
                Change Mode
              </Button>
            </div>

            <Card>
              <CardContent className="p-8">
                <div className="border-2 border-dashed border-neutral-300 rounded-xl p-10 text-center hover:bg-neutral-50 transition-colors">
                  <input
                    type="file"
                    multiple
                    accept="image/*,.pdf"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="bulk-upload"
                  />
                  <label htmlFor="bulk-upload" className="cursor-pointer block">
                    <Upload className="w-16 h-16 text-neutral-400 mx-auto mb-4" />
                    <p className="text-lg font-medium text-dark mb-2">
                      Click to Upload Files
                    </p>
                    <p className="text-sm text-neutral-500">
                      Support for JPG, PNG, PDF
                    </p>
                  </label>
                </div>

                {files.length > 0 && (
                  <div className="mt-6 space-y-3">
                    <h4 className="font-medium text-sm text-neutral-700">
                      Selected Files:
                    </h4>
                    {files.map((f, i) => (
                      <div
                        key={i}
                        className="flex items-center p-3 bg-neutral-50 rounded border"
                      >
                        <FileText className="w-4 h-4 mr-3 text-neutral-500" />
                        <span className="text-sm truncate flex-1">
                          {f.name}
                        </span>
                        <span className="text-xs text-neutral-400 ml-2">
                          {(f.size / 1024 / 1024).toFixed(2)} MB
                        </span>
                      </div>
                    ))}

                    <div className="flex gap-4 mt-4">
                      <Button
                        variant="secondary"
                        onClick={uploadToBucket}
                        loading={uploadingToBucket}
                        disabled={uploadedFileUrls.length > 0}
                      >
                        {uploadedFileUrls.length > 0
                          ? "Uploaded!"
                          : "Upload to Bucket (Optional)"}
                      </Button>
                    </div>
                  </div>
                )}

                <div className="mt-8 pt-6 border-t">
                  <Button
                    size="lg"
                    className="w-full"
                    onClick={submitUploadTest}
                    loading={evaluatingUpload}
                    disabled={files.length === 0}
                  >
                    Evaluate Answer Sheets
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
