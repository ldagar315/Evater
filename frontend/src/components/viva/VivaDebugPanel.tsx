import React, { useState, useEffect } from "react";
import { Bug, ChevronDown, Wifi, Mic, Clock, Activity } from "lucide-react";
import { Button } from "../ui/Button";

interface VivaDebugPanelProps {
  wsRef: React.RefObject<WebSocket | null>;
  wsConnected: boolean;
  isRecording: boolean;
  mediaRecorderRef: React.RefObject<MediaRecorder | null>;
  selectedGrade: string;
  subject: string;
  chapter: string;
  sessionStartTime?: Date;
  questionCount: number;
  answerCount: number;
}

export function VivaDebugPanel({
  wsRef,
  wsConnected,
  isRecording,
  mediaRecorderRef,
  selectedGrade,
  subject,
  chapter,
  sessionStartTime,
  questionCount,
  answerCount,
}: VivaDebugPanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [sessionDuration, setSessionDuration] = useState(0);
  const [messageStats] = useState({ sent: 0, received: 0 });
  const [lastMessageTime] = useState<Date | null>(null);
  const [audioDeviceName, setAudioDeviceName] = useState<string>("N/A");
  const [micPermission, setMicPermission] = useState<
    PermissionState | "unknown"
  >("unknown");
  const [supportedMimeTypes, setSupportedMimeTypes] = useState<string[]>([]);
  const [wsReadyState, setWsReadyState] = useState<number>(-1);
  const [connectionUptime, setConnectionUptime] = useState(0);
  const [lastError] = useState<string | null>(null);

  // Update session duration
  useEffect(() => {
    if (!sessionStartTime) return;

    const interval = setInterval(() => {
      const duration = Math.floor(
        (Date.now() - sessionStartTime.getTime()) / 1000
      );
      setSessionDuration(duration);
    }, 1000);

    return () => clearInterval(interval);
  }, [sessionStartTime]);

  // Update connection uptime
  useEffect(() => {
    if (!wsConnected) {
      setConnectionUptime(0);
      return;
    }

    const startTime = Date.now();
    const interval = setInterval(() => {
      const uptime = Math.floor((Date.now() - startTime) / 1000);
      setConnectionUptime(uptime);
    }, 1000);

    return () => clearInterval(interval);
  }, [wsConnected]);

  // Monitor WebSocket state
  useEffect(() => {
    if (!wsRef.current) {
      setWsReadyState(-1);
      return;
    }

    const updateReadyState = () => {
      if (wsRef.current) {
        setWsReadyState(wsRef.current.readyState);
      }
    };

    updateReadyState();
    const interval = setInterval(updateReadyState, 500);

    return () => clearInterval(interval);
  }, [wsRef, wsConnected]);

  // Check microphone permission
  useEffect(() => {
    if (navigator.permissions) {
      navigator.permissions
        .query({ name: "microphone" as PermissionName })
        .then((permissionStatus) => {
          setMicPermission(permissionStatus.state);
          permissionStatus.onchange = () => {
            setMicPermission(permissionStatus.state);
          };
        })
        .catch(() => setMicPermission("unknown"));
    }
  }, []);

  // Get audio device info
  useEffect(() => {
    navigator.mediaDevices
      .enumerateDevices()
      .then((devices) => {
        const audioInput = devices.find(
          (device) => device.kind === "audioinput"
        );
        if (audioInput) {
          setAudioDeviceName(audioInput.label || "Default Microphone");
        }
      })
      .catch(() => setAudioDeviceName("Error fetching devices"));
  }, []);

  // Check supported MIME types
  useEffect(() => {
    const mimeTypes = [
      "audio/webm",
      "audio/webm;codecs=opus",
      "audio/ogg",
      "audio/mp4",
    ];
    const supported = mimeTypes.filter((type) =>
      MediaRecorder.isTypeSupported(type)
    );
    setSupportedMimeTypes(supported);
  }, []);

  const getReadyStateLabel = (state: number): string => {
    switch (state) {
      case WebSocket.CONNECTING:
        return "CONNECTING";
      case WebSocket.OPEN:
        return "OPEN";
      case WebSocket.CLOSING:
        return "CLOSING";
      case WebSocket.CLOSED:
        return "CLOSED";
      default:
        return "UNKNOWN";
    }
  };

  const getReadyStateColor = (state: number): string => {
    switch (state) {
      case WebSocket.CONNECTING:
        return "text-yellow-600";
      case WebSocket.OPEN:
        return "text-green-600";
      case WebSocket.CLOSING:
        return "text-orange-600";
      case WebSocket.CLOSED:
        return "text-red-600";
      default:
        return "text-gray-600";
    }
  };

  const formatDuration = (seconds: number): string => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    }
    return `${secs}s`;
  };

  const getPermissionColor = (state: PermissionState | "unknown"): string => {
    switch (state) {
      case "granted":
        return "text-green-600";
      case "denied":
        return "text-red-600";
      case "prompt":
        return "text-yellow-600";
      default:
        return "text-gray-600";
    }
  };

  const getPermissionLabel = (state: PermissionState | "unknown"): string => {
    switch (state) {
      case "granted":
        return "Granted";
      case "denied":
        return "Denied";
      case "prompt":
        return "Not Asked";
      default:
        return "Unknown";
    }
  };

  const getRecorderState = (): string => {
    if (!mediaRecorderRef.current) return "Not Initialized";
    return mediaRecorderRef.current.state;
  };

  return (
    <div className="fixed bottom-4 right-4 z-50">
      {/* Toggle Button */}
      {!isOpen && (
        <Button
          onClick={() => setIsOpen(true)}
          className="flex items-center space-x-2 bg-gray-800 hover:bg-gray-700 text-white shadow-lg"
          size="sm"
        >
          <Bug className="h-4 w-4" />
          <span>Debug</span>
        </Button>
      )}

      {/* Debug Panel */}
      {isOpen && (
        <div className="bg-gray-900 text-gray-100 rounded-lg shadow-2xl border border-gray-700 w-96 max-h-[80vh] overflow-hidden flex flex-col">
          {/* Header */}
          <div className="bg-gray-800 px-4 py-3 flex items-center justify-between border-b border-gray-700">
            <div className="flex items-center space-x-2">
              <Bug className="h-5 w-5 text-green-400" />
              <h3 className="font-semibold text-sm">Debug Panel</h3>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="text-gray-400 hover:text-gray-200"
            >
              <ChevronDown className="h-5 w-5" />
            </button>
          </div>

          {/* Content */}
          <div className="overflow-y-auto p-4 space-y-4 text-xs font-mono">
            {/* WebSocket Section */}
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-sm font-bold text-blue-400">
                <Wifi className="h-4 w-4" />
                <span>WebSocket</span>
              </div>

              <div className="bg-gray-800 rounded p-2 space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span
                    className={wsConnected ? "text-green-400" : "text-red-400"}
                  >
                    {wsConnected ? "Connected" : "Disconnected"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Ready State:</span>
                  <span className={getReadyStateColor(wsReadyState)}>
                    {getReadyStateLabel(wsReadyState)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">URL:</span>
                  <span
                    className="text-gray-300 truncate max-w-[200px]"
                    title="wss://ldagar315--evater-v1-wrapper.modal.run/ws/viva"
                  >
                    wss://...modal.run/ws/viva
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Uptime:</span>
                  <span className="text-gray-300">
                    {formatDuration(connectionUptime)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Last Message:</span>
                  <span className="text-gray-300">
                    {lastMessageTime
                      ? lastMessageTime.toLocaleTimeString()
                      : "N/A"}
                  </span>
                </div>
              </div>
            </div>

            {/* Audio Section */}
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-sm font-bold text-purple-400">
                <Mic className="h-4 w-4" />
                <span>Audio System</span>
              </div>

              <div className="bg-gray-800 rounded p-2 space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-400">Permission:</span>
                  <span className={getPermissionColor(micPermission)}>
                    {getPermissionLabel(micPermission)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Recording:</span>
                  <span
                    className={isRecording ? "text-red-400" : "text-gray-400"}
                  >
                    {isRecording ? "Active" : "Inactive"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Recorder State:</span>
                  <span className="text-gray-300">{getRecorderState()}</span>
                </div>
                <div className="flex flex-col space-y-1 mt-2">
                  <span className="text-gray-400">Device:</span>
                  <span className="text-gray-300 text-[10px] break-words">
                    {audioDeviceName}
                  </span>
                </div>
                <div className="flex flex-col space-y-1 mt-2">
                  <span className="text-gray-400">Supported MIME:</span>
                  {supportedMimeTypes.map((type, idx) => (
                    <span key={idx} className="text-green-400 text-[10px]">
                      ✓ {type}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* Session Section */}
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-sm font-bold text-yellow-400">
                <Clock className="h-4 w-4" />
                <span>Session Info</span>
              </div>

              <div className="bg-gray-800 rounded p-2 space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-400">Grade:</span>
                  <span className="text-gray-300">
                    {selectedGrade || "N/A"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Subject:</span>
                  <span className="text-gray-300">{subject || "N/A"}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Chapter:</span>
                  <span
                    className="text-gray-300 truncate max-w-[200px]"
                    title={chapter}
                  >
                    {chapter || "N/A"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Duration:</span>
                  <span className="text-gray-300">
                    {formatDuration(sessionDuration)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Questions:</span>
                  <span className="text-gray-300">{questionCount}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Answers:</span>
                  <span className="text-gray-300">{answerCount}</span>
                </div>
              </div>
            </div>

            {/* Network Stats */}
            <div className="space-y-2">
              <div className="flex items-center space-x-2 text-sm font-bold text-green-400">
                <Activity className="h-4 w-4" />
                <span>Network Stats</span>
              </div>

              <div className="bg-gray-800 rounded p-2 space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-400">Messages Sent:</span>
                  <span className="text-gray-300">{messageStats.sent}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Messages Received:</span>
                  <span className="text-gray-300">{messageStats.received}</span>
                </div>
                {lastError && (
                  <div className="flex flex-col space-y-1 mt-2 pt-2 border-t border-gray-700">
                    <span className="text-red-400">Last Error:</span>
                    <span className="text-red-300 text-[10px] break-words">
                      {lastError}
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="bg-gray-800 px-4 py-2 border-t border-gray-700 text-center">
            <span className="text-[10px] text-gray-500">
              Debug panel - Remove before production
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
