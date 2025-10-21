"use client";

import { useEffect, useState, useRef } from "react";
import { Play, Loader2, Database, Info, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { api, TrainingStatusPoller } from "@/lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import Navigation from "@/components/Navigation";
import { toast } from "sonner";
import Footer from "@/components/Footer";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

type Star = {
  top: string;
  left: string;
  animationDelay: string;
};

type TrainingDatum = {
  epoch: number;
  batch: number;
  loss: number;
  accuracy: number;
};

const TrainPage = () => {
  const [stars, setStars] = useState<Star[]>([]);
  const pollerRef = useRef<TrainingStatusPoller | null>(null);

  useEffect(() => {
    const generatedStars = [...Array(50)].map(() => ({
      top: `${Math.random() * 100}%`,
      left: `${Math.random() * 100}%`,
      animationDelay: `${Math.random() * 3}s`,
    }));
    setStars(generatedStars);
  }, []);

  useEffect(() => {
    const savedData = localStorage.getItem("trainingData");
    if (savedData) {
      try {
        const parsed = JSON.parse(savedData);
        setTrainingData(parsed);
      } catch (e) {
        console.error("Failed to load training data from localStorage");
      }
    }

    const checkTrainingStatus = async () => {
      try {
        const status = await api.getTrainingStatus();

        if (status.is_training) {
          setIsTraining(true);
          setMessage(status.message || "Training in progress...");

          if (status.metrics) {
            setCurrentMetrics({
              loss: status.metrics.loss ?? null,
              accuracy: status.metrics.accuracy ?? null,
              epoch: status.current_epoch,
              totalEpochs: status.total_epochs,
            });
          }

          pollerRef.current = new TrainingStatusPoller((newStatus) => {
            setMessage(newStatus.message);

            if (newStatus.metrics) {
              setCurrentMetrics({
                loss: newStatus.metrics.loss ?? null,
                accuracy: newStatus.metrics.accuracy ?? null,
                epoch: newStatus.current_epoch,
                totalEpochs: newStatus.total_epochs,
              });
            }

            if (newStatus.metrics && newStatus.metrics.loss !== undefined) {
              const batch = newStatus.metrics.batch ?? 0;
              const dataKey = `${newStatus.current_epoch}-${batch}`;

              if (lastDataPointRef.current !== dataKey) {
                lastDataPointRef.current = dataKey;

                setTrainingData((prev) => {
                  const newData = [
                    ...prev,
                    {
                      epoch: newStatus.current_epoch,
                      batch: batch,
                      loss: newStatus.metrics.loss!,
                      accuracy: (newStatus.metrics.accuracy || 0) * 100,
                    },
                  ].slice(-100);

                  return newData;
                });
              }
            }

            if (!newStatus.is_training && newStatus.progress === 100) {
              setIsTraining(false);
              const accuracy = newStatus.metrics.accuracy
                ? (newStatus.metrics.accuracy * 100).toFixed(1)
                : "N/A";
              setMessage(`Training complete! Accuracy: ${accuracy}%`);
              toast.success("Training completed successfully!", {
                description: `Final accuracy: ${accuracy}% • Model ready for predictions`
              });

              if (pollerRef.current) {
                pollerRef.current.stop();
              }
            }
          });

          pollerRef.current.start(2000);
        }
      } catch (error) {
        console.error("Failed to check training status:", error);
      }
    };

    checkTrainingStatus();

    return () => {
      if (pollerRef.current) {
        pollerRef.current.stop();
      }
    };
  }, []);

  const [isTraining, setIsTraining] = useState(false);
  const [trainingData, setTrainingData] = useState<TrainingDatum[]>([]);

  const [hyperparameters, setHyperparameters] = useState({
    learningRate: 0.0003,
    epochs: 30,
    batchSize: 32,
  });

  useEffect(() => {
    if (trainingData.length > 0) {
      localStorage.setItem("trainingData", JSON.stringify(trainingData));
    }
  }, [trainingData]);
  const [message, setMessage] = useState<string>("");
  const [currentMetrics, setCurrentMetrics] = useState<{
    loss: number | null;
    accuracy: number | null;
    epoch: number;
    totalEpochs: number;
  }>({
    loss: null,
    accuracy: null,
    epoch: 0,
    totalEpochs: 0,
  });
  const lastDataPointRef = useRef<string>("");

  const handleTrain = async () => {
    try {
      const status = await api.getTrainingStatus();
      if (status.is_training) {
        toast.error(
          "Training is already in progress! Please wait for it to complete."
        );
        return;
      }
    } catch (error) {
      console.error("Failed to check status:", error);
      toast.error("Cannot connect to backend. Please check your connection.");
      return;
    }

    setIsTraining(true);
    setMessage("Training started...");
    setTrainingData([]);
    localStorage.removeItem("trainingData");
    lastDataPointRef.current = "";
    setCurrentMetrics({
      loss: null,
      accuracy: null,
      epoch: 0,
      totalEpochs: 0,
    });

    toast.loading("Initializing training...", { id: "train-init" });

    try {
      const result = await api.train(
        {
          modelType: "user",
          selectedDatasets: [],
          learningRate: hyperparameters.learningRate,
          epochs: hyperparameters.epochs,
          batchSize: hyperparameters.batchSize,
        },
        undefined
      );

      console.log("Training started:", result);
      setMessage(result.message);
      toast.success("Training started successfully!", {
        id: "train-init",
        description: `${hyperparameters.epochs} epochs with LR ${hyperparameters.learningRate}`
      });

      pollerRef.current = new TrainingStatusPoller((status) => {
        console.log("Training status update:", status);

        setMessage(status.message);

        if (status.metrics) {
          setCurrentMetrics({
            loss: status.metrics.loss ?? null,
            accuracy: status.metrics.accuracy ?? null,
            epoch: status.current_epoch,
            totalEpochs: status.total_epochs,
          });
        }

        if (status.metrics && status.metrics.loss !== undefined) {
          const batch = status.metrics.batch ?? 0;
          const dataKey = `${status.current_epoch}-${batch}`;

          if (lastDataPointRef.current !== dataKey) {
            lastDataPointRef.current = dataKey;

            setTrainingData((prev) => {
              const newData = [
                ...prev,
                {
                  epoch: status.current_epoch,
                  batch: batch,
                  loss: status.metrics.loss!,
                  accuracy: (status.metrics.accuracy || 0) * 100,
                },
              ].slice(-100);

              return newData;
            });
          }
        }

        if (!status.is_training && status.progress === 100) {
          setIsTraining(false);
          const accuracy = status.metrics.accuracy
            ? (status.metrics.accuracy * 100).toFixed(1)
            : "N/A";
          setMessage(`Training complete! Accuracy: ${accuracy}%`);
          toast.success("Training completed successfully!", {
            description: `Final accuracy: ${accuracy}% • Model ready for predictions`
          });
          localStorage.removeItem("trainingData");

          if (pollerRef.current) {
            pollerRef.current.stop();
          }
        }
      });

      pollerRef.current.start(2000);
    } catch (error) {
      console.error("Training error:", error);
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";
      setMessage(`Training failed: ${errorMessage}`);
      toast.error("Training failed", {
        description: errorMessage
      });
      setIsTraining(false);
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

      <div className="container mx-auto px-4 pt-32 pb-16 space-y-8">
        <div className="text-center space-y-4 animate-slide-in">
          <h1 className="font-display font-bold text-4xl sm:text-5xl">
            Train Your <span className="gradient-text">AI Model</span>
          </h1>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Upload training data from NASA missions or select datasets to
            configure your model
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6">
          <Card className="bg-card/50 backdrop-blur border-border/40">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="w-5 h-5 text-primary" /> Training
                Information
              </CardTitle>
              <CardDescription>
                Train your custom model with NASA exoplanet datasets
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 bg-primary/10 border border-primary/20 rounded-lg">
                <p className="text-sm font-medium mb-3">
                  Training with NASA Datasets:
                </p>
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="w-2 h-2 rounded-full bg-primary" />
                    <span>Kepler Objects of Interest</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="w-2 h-2 rounded-full bg-primary" />
                    <span>K2 Mission Candidates</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <div className="w-2 h-2 rounded-full bg-primary" />
                    <span>TESS Exoplanet Candidates</span>
                  </div>
                </div>
              </div>

              <div className="p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                <p className="text-sm font-medium text-blue-500 mb-2">
                  Custom Model Training
                </p>
                <p className="text-xs text-muted-foreground">
                  This will train your personal custom model. After training
                  completes, you can use it for predictions by selecting
                  &quot;User Custom Model&quot; on the Predict page.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-card/50 backdrop-blur border-border/40">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5 text-primary" /> Training
                Configuration
              </CardTitle>
              <CardDescription>
                Customize hyperparameters for model training
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="space-y-2">
                  <Label htmlFor="learningRate" className="text-sm font-medium">
                    Learning Rate
                  </Label>
                  <Input
                    id="learningRate"
                    type="number"
                    step="0.0001"
                    min="0.0001"
                    max="0.01"
                    value={hyperparameters.learningRate}
                    onChange={(e) => {
                      const value = parseFloat(e.target.value);
                      setHyperparameters({
                        ...hyperparameters,
                        learningRate: value,
                      });
                      if (value < 0.0001 || value > 0.01) {
                        toast.warning("Learning rate outside recommended range", {
                          description: "Recommended range: 0.0001 - 0.01"
                        });
                      }
                    }}
                    disabled={isTraining}
                    className="bg-card/50"
                  />
                  <p className="text-xs text-muted-foreground">
                    Default: 0.0003 (Range: 0.0001 - 0.01)
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="epochs" className="text-sm font-medium">
                    Epochs
                  </Label>
                  <Input
                    id="epochs"
                    type="number"
                    min="5"
                    max="100"
                    value={hyperparameters.epochs}
                    onChange={(e) => {
                      const value = parseInt(e.target.value);
                      setHyperparameters({
                        ...hyperparameters,
                        epochs: value,
                      });
                      if (value < 5 || value > 100) {
                        toast.warning("Epochs outside recommended range", {
                          description: "Recommended range: 5 - 100"
                        });
                      }
                    }}
                    disabled={isTraining}
                    className="bg-card/50"
                  />
                  <p className="text-xs text-muted-foreground">
                    Default: 30 (Range: 5 - 100)
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="batchSize" className="text-sm font-medium">
                    Batch Size
                  </Label>
                  <Input
                    id="batchSize"
                    type="number"
                    min="8"
                    max="128"
                    step="8"
                    value={hyperparameters.batchSize}
                    onChange={(e) => {
                      const value = parseInt(e.target.value);
                      setHyperparameters({
                        ...hyperparameters,
                        batchSize: value,
                      });
                      if (value < 8 || value > 128) {
                        toast.warning("Batch size outside recommended range", {
                          description: "Recommended range: 8 - 128"
                        });
                      }
                    }}
                    disabled={isTraining}
                    className="bg-card/50"
                  />
                  <p className="text-xs text-muted-foreground">
                    Default: 32 (Range: 8 - 128)
                  </p>
                </div>
              </div>

              <div className="mt-4 p-3 bg-muted/50 rounded-lg">
                <p className="text-xs text-muted-foreground">
                  <strong>💡 Tips:</strong> Higher learning rates train faster
                  but may be unstable. More epochs improve accuracy but take
                  longer. Larger batch sizes are faster but use more memory.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card className="bg-card/50 backdrop-blur border-border/40">
          <CardContent className="pt-6 space-y-4">
            <Button
              size="lg"
              onClick={handleTrain}
              disabled={isTraining}
              className="w-full bg-primary hover:bg-primary/90 text-primary-foreground glow-electric"
            >
              {isTraining ? (
                <>
                  <Loader2 className="mr-2 w-5 h-5 animate-spin" />
                  Training Model...
                </>
              ) : (
                <>
                  <Play className="mr-2 w-5 h-5" />
                  Start Training
                </>
              )}
            </Button>

            {isTraining && (
              <div className="flex items-start gap-3 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                <Info className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
                <div className="text-sm space-y-1">
                  <p className="font-medium text-blue-500">
                    Training in Progress
                  </p>
                  <p className="text-muted-foreground">
                    Training may take several hours on CPU. You can safely close
                    this tab - training will continue on the server. Return to
                    this page to check progress.
                  </p>
                </div>
              </div>
            )}

            {message && (
              <p className="text-sm text-muted-foreground text-center mt-2">
                {message}
              </p>
            )}

            {isTraining && (
              <div className="mt-4 space-y-4">
                {currentMetrics.loss !== null && (
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-card border border-border rounded-lg">
                      <p className="text-xs text-muted-foreground mb-1">Loss</p>
                      <p className="text-2xl font-bold text-primary">
                        {currentMetrics.loss.toFixed(4)}
                      </p>
                    </div>
                    <div className="p-4 bg-card border border-border rounded-lg">
                      <p className="text-xs text-muted-foreground mb-1">
                        Accuracy
                      </p>
                      <p className="text-2xl font-bold text-primary">
                        {currentMetrics.accuracy !== null
                          ? `${(currentMetrics.accuracy * 100).toFixed(1)}%`
                          : "N/A"}
                      </p>
                    </div>
                  </div>
                )}

                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="text-primary font-medium">
                      {currentMetrics.epoch > 0
                        ? `Epoch ${currentMetrics.epoch}/${currentMetrics.totalEpochs}`
                        : "Training..."}
                    </span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{
                        width:
                          currentMetrics.totalEpochs > 0
                            ? `${
                                (currentMetrics.epoch /
                                  currentMetrics.totalEpochs) *
                                100
                              }%`
                            : "0%",
                      }}
                    />
                  </div>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        {trainingData.length > 0 && (
          <Card className="bg-card/50 backdrop-blur border-border/40">
            <CardHeader>
              <CardTitle>Training Metrics</CardTitle>
              <CardDescription>
                Real-time visualization of model performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex gap-4 text-sm">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-destructive rounded-full"></div>
                    <span className="text-muted-foreground">Loss</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 bg-primary rounded-full"></div>
                    <span className="text-muted-foreground">Accuracy</span>
                  </div>
                  <div className="ml-auto text-muted-foreground">
                    {trainingData.length} data points
                  </div>
                </div>
                <ResponsiveContainer width="100%" height={350}>
                  <LineChart
                    data={trainingData}
                    margin={{ top: 5, right: 30, left: 20, bottom: 25 }}
                  >
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="var(--color-border)"
                      opacity={0.3}
                    />
                    <XAxis
                      dataKey="batch"
                      stroke="var(--color-muted-foreground)"
                      label={{
                        value: "Batch",
                        position: "insideBottom",
                        offset: -15,
                        style: {
                          fill: "var(--color-foreground)",
                          fontWeight: 500,
                        },
                      }}
                      tick={{ fill: "var(--color-foreground)", fontSize: 12 }}
                    />
                    <YAxis
                      yAxisId="left"
                      stroke="var(--color-destructive)"
                      label={{
                        value: "Loss",
                        angle: -90,
                        position: "insideLeft",
                        style: {
                          fill: "var(--color-foreground)",
                          fontWeight: 500,
                        },
                      }}
                      tick={{ fill: "var(--color-foreground)", fontSize: 12 }}
                      domain={[0, "auto"]}
                    />
                    <YAxis
                      yAxisId="right"
                      orientation="right"
                      stroke="var(--color-primary)"
                      label={{
                        value: "Accuracy (%)",
                        angle: 90,
                        position: "insideRight",
                        style: {
                          fill: "var(--color-foreground)",
                          fontWeight: 500,
                        },
                      }}
                      tick={{ fill: "var(--color-foreground)", fontSize: 12 }}
                      domain={[0, 100]}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "var(--color-card)",
                        border: "1px solid var(--color-border)",
                        borderRadius: "8px",
                        padding: "8px 12px",
                        boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
                      }}
                      itemStyle={{ color: "var(--color-foreground)" }}
                      labelStyle={{
                        color: "var(--color-foreground)",
                        fontWeight: "bold",
                      }}
                      formatter={(value: number | string, name: string) => {
                        const numValue =
                          typeof value === "number" ? value : parseFloat(value);
                        if (name === "Loss") {
                          return [numValue.toFixed(4), name];
                        }
                        return [numValue.toFixed(2) + "%", name];
                      }}
                      labelFormatter={(value, payload) => {
                        if (payload && payload[0]) {
                          const data = payload[0].payload;
                          return `Epoch ${data.epoch}, Batch ${value}`;
                        }
                        return `Batch ${value}`;
                      }}
                    />
                    <Line
                      yAxisId="left"
                      type="monotone"
                      dataKey="loss"
                      stroke="var(--color-destructive)"
                      name="Loss"
                      strokeWidth={2.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                    <Line
                      yAxisId="right"
                      type="monotone"
                      dataKey="accuracy"
                      stroke="var(--color-primary)"
                      name="Accuracy"
                      strokeWidth={2.5}
                      dot={false}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
      <Footer />
    </div>
  );
};

export default TrainPage;
