import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Mic,
  MicOff,
  MessageSquare,
  Brain,
  AlertCircle,
  X,
} from "lucide-react";
import { Button } from "../components/ui/Button";
import { Select } from "../components/ui/Select";
import { Card, CardContent, CardHeader } from "../components/ui/Card";
import { Header } from "../components/layout/Header";
import { VivaDebugPanel } from "../components/viva/VivaDebugPanel";
import { useAuthContext } from "../contexts/AuthContext";
import { supabase } from "../lib/supabase";
import { ChapterContent } from "../types";

interface VivaQuestion {
  question: string;
}

interface VivaMessage {
  type: "question" | "status" | "error" | "feedback";
  content: string;
  timestamp: Date;
}

interface ConceptScore {
  correctness: number;
  depth: number;
  clarity: number;
}

interface VivaResults {
  scores: Record<string, ConceptScore>;
  feedback: string[];
}

export function VivaPage() {
  const navigate = useNavigate();
  const { user } = useAuthContext();

  // Form state
  const [selectedGrade, setSelectedGrade] = useState("");
  const [subject, setSubject] = useState("");
  const [chapter, setChapter] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Chapter data
  const [chapterContents, setChapterContents] = useState<ChapterContent[]>([]);
  const [availableSubjects, setAvailableSubjects] = useState<string[]>([]);
  const [availableChapters, setAvailableChapters] = useState<string[]>([]);

  // Viva session state
  const [vivaStarted, setVivaStarted] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState("");
  const [messages, setMessages] = useState<VivaMessage[]>([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [vivaResults, setVivaResults] = useState<VivaResults | null>(null);
  const [sessionCompleted, setSessionCompleted] = useState(false);
  const [sessionStartTime, setSessionStartTime] = useState<Date | null>(null);
  const [questionCount, setQuestionCount] = useState(0);
  const [answerCount, setAnswerCount] = useState(0);
  const [currentFeedback, setCurrentFeedback] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [isFeedbackExpanded, setIsFeedbackExpanded] = useState(true);

  // WebSocket and audio refs
  const wsRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const gradeOptions = Array.from({ length: 12 }, (_, i) => ({
    value: (i + 1).toString(),
    label: `Grade ${i + 1}`,
  }));

  useEffect(() => {
    fetchChapterContents();
    return () => {
      // Cleanup WebSocket on unmount
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
    };
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

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const fetchChapterContents = async () => {
    try {
      const { data, error } = await supabase
        .from("Chapter_contents")
        .select("*");

      if (error) throw error;
      setChapterContents(data || []);
    } catch (err) {
      console.error("Error fetching chapter contents:", err);
      setError("Failed to load chapter contents");
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const addMessage = (type: VivaMessage["type"], content: string) => {
    setMessages((prev) => [
      ...prev,
      {
        type,
        content,
        timestamp: new Date(),
      },
    ]);
  };

  const initializeWebSocket = () => {
    // Always use the external WebSocket endpoint
    const wsUrl = "wss://ldagar315--evater-v1-wrapper.modal.run/ws/viva";

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log("WebSocket connected");
      setWsConnected(true);
      addMessage("status", "Connected to Viva session");

      // Start ping interval
      if (pingIntervalRef.current) clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, 30000);
    };

    ws.onmessage = (event) => {
      try {
        // Check if the message is JSON or binary
        if (typeof event.data === "string") {
          const data = JSON.parse(event.data);
          console.log("Received from WebSocket:", data);

          if (data.status === "connected") {
            addMessage("status", data.message);
            // Send chapter information after receiving connected status
            const chapterInfo = {
              grade: selectedGrade,
              subject: subject,
              chapter: chapter,
            };
            console.log("Sending chapter info:", chapterInfo);
            ws.send(JSON.stringify(chapterInfo));
          } else if (data.question) {
            setCurrentQuestion(data.question);
            setQuestionCount((prev) => prev + 1);
            // Don't clear currentFeedback here, just collapse it
            setIsFeedbackExpanded(false);
            setIsEvaluating(false);
            addMessage("question", data.question);
          } else if (data.answer) {
            setCurrentFeedback(data.answer);
            setIsFeedbackExpanded(true);
            setIsEvaluating(false);
            addMessage("feedback", data.answer);
          } else if (data.feedback) {
            // Final session results
            console.log("Received final feedback:", data.feedback);
            setVivaResults(data.feedback);
            setSessionCompleted(true);
            addMessage(
              "status",
              "Viva session completed! View your results below."
            );
          }
        }
      } catch (err) {
        console.error("Error parsing WebSocket message:", err);
        addMessage("error", "Error processing server response");
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      addMessage("error", "Connection error occurred");
      setWsConnected(false);
      setIsEvaluating(false);
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
      setWsConnected(false);
      addMessage("status", "Viva session ended");
      setIsEvaluating(false);
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
        pingIntervalRef.current = null;
      }
    };
  };

  const startViva = async () => {
    if (!selectedGrade || !subject || !chapter) {
      setError("Please fill in all required fields");
      return;
    }

    setLoading(true);
    setError("");

    try {
      // Request microphone permission
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop()); // Stop the test stream

      setVivaStarted(true);
      setMessages([]);
      setSessionStartTime(new Date());
      setQuestionCount(0);
      setAnswerCount(0);
      setCurrentFeedback(null);
      setShowHistory(false);
      initializeWebSocket();
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setError("Microphone access is required for the viva session");
    } finally {
      setLoading(false);
    }
  };

  const startRecording = async () => {
    if (!wsConnected || !wsRef.current) {
      addMessage("error", "Not connected to server");
      return;
    }

    // Clear feedback when starting new recording
    setCurrentFeedback(null);
    setIsFeedbackExpanded(false);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "audio/webm",
      });

      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: "audio/webm",
        });

        // Send audio data to WebSocket
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.send(audioBlob);
          setAnswerCount((prev) => prev + 1);
          addMessage("status", "Answer submitted, processing...");
          setIsEvaluating(true); // Start evaluating state
        }

        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
      addMessage("status", "Recording your answer...");
    } catch (err) {
      console.error("Error starting recording:", err);
      addMessage("error", "Failed to start recording");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const endViva = () => {
    if (wsRef.current) {
      wsRef.current.close();
    }
    setVivaStarted(false);
    setWsConnected(false);
    setCurrentQuestion("");
    setMessages([]);
    setVivaResults(null);
    setSessionCompleted(false);
    setSessionStartTime(null);
    setQuestionCount(0);
    setAnswerCount(0);
    setCurrentFeedback(null);
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString("en-US", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  if (!vivaStarted) {
    return (
      <div className="min-h-screen bg-cream">
        <Header />

        <div className="max-w-2xl mx-auto py-4 sm:py-8 px-4 sm:px-6 lg:px-8">
          <div className="mb-4 sm:mb-6">
            <Button
              variant="ghost"
              onClick={() => navigate("/home")}
              className="flex items-center"
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Home
            </Button>
          </div>

          <Card>
            <CardHeader>
              <div className="flex items-center mb-2">
                <Brain className="h-6 w-6 text-primary-600 mr-2" />
                <h2 className="text-xl sm:text-2xl font-bold text-dark">
                  AI Viva Session
                </h2>
              </div>
              <p className="text-sm text-neutral-600">
                Start an interactive viva session with AI. Select your subject
                and chapter to begin.
              </p>
            </CardHeader>

            <CardContent>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  startViva();
                }}
                className="space-y-4 sm:space-y-6"
              >
                <Select
                  label="Grade"
                  value={selectedGrade}
                  onChange={(e) => setSelectedGrade(e.target.value)}
                  options={[
                    { value: "", label: "Select grade" },
                    ...gradeOptions,
                  ]}
                  required
                />

                <Select
                  label="Subject"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  options={[
                    { value: "", label: "Select subject" },
                    ...availableSubjects.map((s) => ({ value: s, label: s })),
                  ]}
                  required
                  disabled={!selectedGrade}
                />

                <Select
                  label="Chapter"
                  value={chapter}
                  onChange={(e) => setChapter(e.target.value)}
                  options={[
                    { value: "", label: "Select chapter" },
                    ...availableChapters.map((c) => ({ value: c, label: c })),
                  ]}
                  required
                  disabled={!subject}
                />

                {error && (
                  <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                    <p className="text-sm text-red-600">{error}</p>
                  </div>
                )}

                <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 sm:p-4">
                  <h3 className="text-sm font-medium text-blue-900 mb-2">
                    What to Expect:
                  </h3>
                  <ul className="text-sm text-blue-800 space-y-1">
                    <li>
                      • AI will ask questions based on your selected chapter
                    </li>
                    <li>• Speak your answers clearly into the microphone</li>
                    <li>• Get real-time feedback and follow-up questions</li>
                    <li>• Session adapts based on your responses</li>
                  </ul>
                </div>

                <Button
                  type="submit"
                  className="w-full flex items-center justify-center h-12 sm:h-14 text-base sm:text-lg"
                  loading={loading}
                  disabled={!selectedGrade || !subject || !chapter}
                >
                  {loading ? (
                    "Starting Viva Session..."
                  ) : (
                    <>
                      <Brain className="h-5 w-5 mr-2" />
                      Start Viva Session
                    </>
                  )}
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-cream flex flex-col z-50 overflow-hidden font-sans">
      {/* Immersive Session Header */}
      <div className="flex-none bg-white/80 backdrop-blur-md border-b border-neutral-100 px-4 py-3 flex items-center justify-between z-20">
        <div className="flex items-center space-x-3">
          <div
            className={`flex items-center px-2 py-1 rounded-full text-xs font-medium ${
              wsConnected
                ? "bg-green-100 text-green-700"
                : "bg-red-100 text-red-700"
            }`}
          >
            <div
              className={`w-1.5 h-1.5 rounded-full mr-1.5 ${
                wsConnected ? "bg-green-500 animate-pulse" : "bg-red-500"
              }`}
            />
            {wsConnected ? "Live" : "Offline"}
          </div>
          <div className="h-4 w-px bg-neutral-200" />
          <div className="flex flex-col">
            <span className="text-xs font-semibold text-neutral-500 uppercase tracking-wider">
              {subject}
            </span>
            <span className="text-xs text-neutral-400 truncate max-w-[150px] sm:max-w-xs">
              {chapter}
            </span>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowHistory(!showHistory)}
            className={`text-neutral-500 hover:text-primary-600 ${
              showHistory ? "bg-primary-50 text-primary-600" : ""
            }`}
          >
            <MessageSquare className="h-5 w-5" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={endViva}
            className="text-neutral-400 hover:text-red-600 hover:bg-red-50"
          >
            <X className="h-5 w-5" />
          </Button>
        </div>
      </div>

      {/* Main Content Area - Scrollable */}
      <div className="flex-1 overflow-y-auto p-4 pb-32 sm:pb-4 scroll-smooth">
        <div className="max-w-3xl mx-auto w-full space-y-6 pt-4 sm:pt-10">
          {/* Question Card */}
          {currentQuestion && (
            <div className="animate-fade-in">
              <div className="bg-white rounded-3xl p-6 sm:p-10 shadow-sm border border-neutral-100 relative overflow-hidden group">
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary-400 to-primary-600" />
                <div className="flex flex-col gap-4">
                  <span className="inline-flex self-start items-center px-3 py-1 rounded-full bg-primary-50 text-primary-700 text-xs font-bold uppercase tracking-wider">
                    Question {questionCount}
                  </span>
                  <h3 className="text-2xl sm:text-3xl md:text-4xl font-bold text-dark leading-tight tracking-tight">
                    {currentQuestion}
                  </h3>
                </div>
              </div>
            </div>
          )}

          {/* Collapsible Feedback Card */}
          {currentFeedback && (
            <div className="animate-slide-up transition-all duration-300 ease-in-out">
              {isFeedbackExpanded ? (
                <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-2xl p-5 sm:p-8 border border-green-100 shadow-sm relative">
                  <button
                    onClick={() => setIsFeedbackExpanded(false)}
                    className="absolute top-4 right-4 text-green-700 hover:text-green-900 p-1 rounded-full hover:bg-green-100 transition-colors"
                  >
                    <X className="h-4 w-4" />
                  </button>
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-xl bg-green-500 flex items-center justify-center flex-shrink-0 shadow-sm text-white">
                      <Brain className="h-6 w-6" />
                    </div>
                    <div className="space-y-2">
                      <h4 className="text-sm font-bold text-green-800 uppercase tracking-wide">
                        Feedback
                      </h4>
                      <p className="text-base sm:text-lg text-green-900 leading-relaxed">
                        {currentFeedback}
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <button
                  onClick={() => setIsFeedbackExpanded(true)}
                  className="w-full bg-white border border-green-100 rounded-xl p-4 shadow-sm flex items-center justify-between hover:bg-green-50 transition-colors group"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-green-100 flex items-center justify-center text-green-600 group-hover:bg-green-200 transition-colors">
                      <Brain className="h-4 w-4" />
                    </div>
                    <span className="text-sm font-medium text-green-800">
                      View Feedback for Question {questionCount - 1}
                    </span>
                  </div>
                  <div className="text-green-400">
                    <ArrowLeft className="h-4 w-4 rotate-[-90deg]" />
                  </div>
                </button>
              )}
            </div>
          )}

          {/* Empty State / Loading */}
          {!currentQuestion && wsConnected && (
            <div className="flex flex-col items-center justify-center py-20 text-neutral-400 animate-pulse">
              <Brain className="h-12 w-12 mb-4 opacity-50" />
              <p className="text-lg font-medium">Preparing your session...</p>
            </div>
          )}
        </div>
      </div>

      {/* History Drawer (Mobile) / Panel (Desktop) */}
      {showHistory && (
        <div className="absolute inset-0 z-30 bg-cream/95 backdrop-blur-sm sm:static sm:bg-transparent sm:backdrop-blur-none sm:border-l sm:border-neutral-100 sm:w-80 sm:flex-none transition-all duration-300">
          <div className="h-full flex flex-col p-4 sm:p-0">
            <div className="flex items-center justify-between mb-4 sm:hidden">
              <h3 className="text-lg font-bold text-dark">Session History</h3>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowHistory(false)}
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
            <div className="flex-1 overflow-y-auto space-y-4 sm:p-4">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-xl text-sm ${
                    message.type === "question"
                      ? "bg-white border border-neutral-100 shadow-sm"
                      : message.type === "feedback"
                      ? "bg-green-50 border border-green-100"
                      : "bg-neutral-50"
                  }`}
                >
                  <div className="flex justify-between items-center mb-1">
                    <span
                      className={`text-xs font-bold uppercase ${
                        message.type === "question"
                          ? "text-primary-600"
                          : message.type === "feedback"
                          ? "text-green-600"
                          : "text-neutral-500"
                      }`}
                    >
                      {message.type}
                    </span>
                    <span className="text-[10px] text-neutral-400">
                      {formatTime(message.timestamp)}
                    </span>
                  </div>
                  <p className="text-dark leading-relaxed">{message.content}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Bottom Action Bar (FAB) */}
      <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-cream via-cream to-transparent z-40 pointer-events-none flex justify-center">
        <div className="pointer-events-auto flex items-center gap-6">
          {isEvaluating ? (
            <div className="flex flex-col items-center gap-3 animate-pulse">
              <div className="w-20 h-20 rounded-full bg-primary-100 flex items-center justify-center border-4 border-primary-200">
                <Brain className="h-8 w-8 text-primary-600 animate-bounce" />
              </div>
              <span className="text-sm font-bold text-primary-700 bg-white/80 backdrop-blur px-3 py-1 rounded-full shadow-sm">
                Evaluating...
              </span>
            </div>
          ) : !isRecording ? (
            <button
              onClick={startRecording}
              disabled={!wsConnected || !currentQuestion}
              className="group relative flex items-center justify-center w-20 h-20 rounded-full bg-dark text-white shadow-2xl shadow-neutral-400 hover:scale-105 active:scale-95 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="absolute inset-0 rounded-full bg-white/10 group-hover:scale-110 transition-transform duration-500" />
              <Mic className="h-8 w-8" />
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="group relative flex items-center justify-center w-20 h-20 rounded-full bg-red-500 text-white shadow-2xl shadow-red-200 hover:scale-105 active:scale-95 transition-all duration-300"
            >
              <div className="absolute inset-0 rounded-full border-4 border-red-200 animate-ping" />
              <div className="w-8 h-8 bg-white rounded-lg" />
            </button>
          )}
        </div>
      </div>

      {/* Debug Panel */}
      <VivaDebugPanel
        wsRef={wsRef}
        wsConnected={wsConnected}
        isRecording={isRecording}
        mediaRecorderRef={mediaRecorderRef}
        selectedGrade={selectedGrade}
        subject={subject}
        chapter={chapter}
        sessionStartTime={sessionStartTime || undefined}
        questionCount={questionCount}
        answerCount={answerCount}
      />
    </div>
  );
}
