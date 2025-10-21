"use client";

import { useEffect, useState, useRef } from "react";
import {
  Terminal,
  RefreshCw,
  Trash2,
  Download,
  Play,
  Pause,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";
import { Badge } from "@/components/ui/badge";
import { toast } from "sonner";

type LogEntry = {
  timestamp: string;
  level: "INFO" | "ERROR" | "WARNING" | "DEBUG";
  message: string;
};

type Star = {
  top: string;
  left: string;
  animationDelay: string;
};

const ConsolePage = () => {
  const [stars, setStars] = useState<Star[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isAutoRefresh, setIsAutoRefresh] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const logsEndRef = useRef<HTMLDivElement>(null);
  const autoRefreshInterval = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    const generatedStars = [...Array(50)].map(() => ({
      top: `${Math.random() * 100}%`,
      left: `${Math.random() * 100}%`,
      animationDelay: `${Math.random() * 3}s`,
    }));
    setStars(generatedStars);
  }, []);

  useEffect(() => {
    if (logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  const fetchLogs = async () => {
    try {
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/logs`);
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.logs) {
          setLogs(data.logs.slice(-100));
        }
      }
      setLastUpdate(new Date());
    } catch (error) {
      console.error("Failed to fetch logs:", error);
      toast.error("Failed to fetch logs", {
        description: "Could not connect to backend API",
      });
      const mockLogs: LogEntry[] = [
        {
          timestamp: new Date().toISOString(),
          level: "INFO",
          message: "Server started on port 8000",
        },
        {
          timestamp: new Date().toISOString(),
          level: "INFO",
          message: "Connected to Railway deployment",
        },
        {
          timestamp: new Date().toISOString(),
          level: "WARNING",
          message: "Could not connect to backend logs API",
        },
      ];
      setLogs(mockLogs);
      setLastUpdate(new Date());
    }
  };

  useEffect(() => {
    if (isAutoRefresh) {
      fetchLogs();
      autoRefreshInterval.current = setInterval(fetchLogs, 5000);
    } else {
      if (autoRefreshInterval.current) {
        clearInterval(autoRefreshInterval.current);
      }
    }

    return () => {
      if (autoRefreshInterval.current) {
        clearInterval(autoRefreshInterval.current);
      }
    };
  }, [isAutoRefresh]);

  const clearLogs = async () => {
    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_URL}/logs/clear`,
        {
          method: "POST",
        }
      );
      if (response.ok) {
        setLogs([]);
        toast.success("Logs cleared successfully");
        fetchLogs();
      }
    } catch (error) {
      console.error("Failed to clear logs:", error);
      toast.error("Failed to clear logs", {
        description: "Could not connect to backend API",
      });
      setLogs([]);
    }
  };

  const downloadLogs = () => {
    try {
      const logsText = logs
        .map((log) => `[${log.timestamp}] ${log.level}: ${log.message}`)
        .join("\n");

      const blob = new Blob([logsText], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const filename = `railway-logs-${new Date().toISOString()}.txt`;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      toast.success("Logs downloaded", {
        description: `${logs.length} log entries saved to ${filename}`,
      });
    } catch (error) {
      console.error("Failed to download logs:", error);
      toast.error("Failed to download logs");
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case "ERROR":
        return "text-red-500";
      case "WARNING":
        return "text-yellow-500";
      case "INFO":
        return "text-blue-500";
      case "DEBUG":
        return "text-gray-500";
      default:
        return "text-foreground";
    }
  };

  const getLevelBadgeVariant = (
    level: string
  ): "default" | "destructive" | "outline" | "secondary" => {
    switch (level) {
      case "ERROR":
        return "destructive";
      case "WARNING":
        return "outline";
      case "INFO":
        return "default";
      case "DEBUG":
        return "secondary";
      default:
        return "outline";
    }
  };

  return (
    <div className="min-h-screen cosmic-bg">
      <Navigation />
      <div className="absolute inset-0 cosmic-glow opacity-50" />
      <div className="absolute inset-0">
        {stars.map((star, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-primary/30 rounded-full animate-pulse-glow"
            style={star}
          />
        ))}
      </div>

      <div className="container mx-auto px-4 pt-32 pb-16">
        <div className="max-w-6xl mx-auto space-y-6">
          <div className="text-center space-y-4 animate-slide-in">
            <h1 className="font-display font-bold text-4xl sm:text-5xl">
              Server <span className="gradient-text">Console</span>
            </h1>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              Monitor Railway backend logs in real-time
            </p>
          </div>

          <Card className="bg-card/50 backdrop-blur border-border/40">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <Terminal className="w-5 h-5 text-primary" />
                    Live Logs
                  </CardTitle>
                  <CardDescription>
                    Last updated: {lastUpdate.toLocaleTimeString()}
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      const newState = !isAutoRefresh;
                      setIsAutoRefresh(newState);
                      toast.info(
                        newState
                          ? "Auto-refresh enabled"
                          : "Auto-refresh paused",
                        {
                          description: newState
                            ? "Logs will update every 5 seconds"
                            : "Click Resume to continue monitoring",
                        }
                      );
                    }}
                  >
                    {isAutoRefresh ? (
                      <>
                        <Pause className="w-4 h-4 mr-2" />
                        Pause
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 mr-2" />
                        Resume
                      </>
                    )}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      fetchLogs();
                      toast.info("Refreshing logs...");
                    }}
                    disabled={isAutoRefresh}
                  >
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Refresh
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={downloadLogs}
                    disabled={logs.length === 0}
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={clearLogs}
                    disabled={logs.length === 0}
                  >
                    <Trash2 className="w-4 h-4 mr-2" />
                    Clear
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="bg-black/90 rounded-lg p-4 font-mono text-sm h-[600px] overflow-y-auto border border-border/40">
                {logs.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-muted-foreground">
                    <div className="text-center space-y-2">
                      <Terminal className="w-12 h-12 mx-auto opacity-50" />
                      <p>No logs yet. Waiting for server output...</p>
                      {!isAutoRefresh && (
                        <p className="text-xs">
                          Auto-refresh is paused. Click Resume to start
                          monitoring.
                        </p>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="space-y-1">
                    {logs.map((log, index) => (
                      <div
                        key={index}
                        className="flex items-start gap-3 py-1 hover:bg-white/5 rounded px-2 transition-colors"
                      >
                        <span className="text-gray-500 text-xs whitespace-nowrap">
                          {new Date(log.timestamp).toLocaleTimeString()}
                        </span>
                        <Badge
                          variant={getLevelBadgeVariant(log.level)}
                          className="text-xs font-mono w-16 justify-center"
                        >
                          {log.level}
                        </Badge>
                        <span className={`flex-1 ${getLevelColor(log.level)}`}>
                          {log.message}
                        </span>
                      </div>
                    ))}
                    <div ref={logsEndRef} />
                  </div>
                )}
              </div>

              {isAutoRefresh && (
                <div className="mt-3 flex items-center gap-2 text-xs text-muted-foreground">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                  Auto-refreshing every 5 seconds
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default ConsolePage;
