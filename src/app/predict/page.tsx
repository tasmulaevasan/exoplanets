"use client";

import { useEffect, useState } from "react";
import { Upload, Download, Sparkles, Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import Footer from "@/components/Footer";
import { api } from "@/lib/api";
import type { ModelInfo } from "@/lib/api";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import Navigation from "@/components/Navigation";
import ModelSelector from "@/components/ModelSelector";
import { toast } from "sonner";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

type Star = {
  top: string;
  left: string;
  animationDelay: string;
};

type ManualParams = {
  period: string;
  duration: string;
  depth: string;
  stellarRadius: string;
  stellarTemp: string;
};

type Prediction = {
  id: string;
  prediction: string;
  confidence: number;
  period: string;
  radius: string;
};

const PredictPage = () => {
  const [stars, setStars] = useState<Star[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>("pretrained");

  useEffect(() => {
    const generatedStars = [...Array(50)].map(() => ({
      top: `${Math.random() * 100}%`,
      left: `${Math.random() * 100}%`,
      animationDelay: `${Math.random() * 3}s`,
    }));
    setStars(generatedStars);

    const fetchModelInfo = async () => {
      try {
        const info = await api.getModelInfo();
        setModelInfo(info);

        const models = await api.getModels();
        const pretrainedModel = models.models.find(
          (m) => m.type === "pretrained"
        );
        const userModel = models.models.find((m) => m.type === "user");

        if (pretrainedModel?.available) {
          setSelectedModel("pretrained");
        } else if (userModel?.available) {
          setSelectedModel("user");
        }
      } catch (error) {
        console.error("Failed to fetch model info:", error);
      }
    };
    fetchModelInfo();
  }, []);

  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [manualParams, setManualParams] = useState<ManualParams>({
    period: "",
    duration: "",
    depth: "",
    stellarRadius: "",
    stellarTemp: "",
  });
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [isPredicting, setIsPredicting] = useState(false);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploadedFile(file);
      toast.success(`File uploaded: ${file.name}`, {
        description: "Ready for prediction"
      });
    }
  };

  const handlePredict = async () => {
    if (!uploadedFile && !manualParams.period) {
      toast.error("No data provided", {
        description: "Please upload a CSV or enter manual parameters."
      });
      return;
    }

    setIsPredicting(true);
    toast.loading("Running prediction...", { id: "prediction" });

    try {
      if (uploadedFile) {
        const result = await api.predictCSV(uploadedFile, selectedModel);
        setPredictions(result.predictions);
        toast.success(
          `Predictions complete! Analyzed ${result.predictions.length} candidates`,
          {
            id: "prediction",
            description: `Using ${selectedModel} model`
          }
        );
      } else {
        const result = await api.predictManual(
          {
            period: parseFloat(manualParams.period),
            duration: parseFloat(manualParams.duration),
            depth: parseFloat(manualParams.depth),
            stellarRadius: parseFloat(manualParams.stellarRadius),
            stellarTemp: parseFloat(manualParams.stellarTemp),
          },
          selectedModel
        );
        setPredictions(result.predictions);

        const prediction = result.predictions[0].prediction;
        const confidence = (result.predictions[0].confidence * 100).toFixed(1);

        toast.success(
          `Prediction: ${prediction}`,
          {
            id: "prediction",
            description: `Confidence: ${confidence}% (${selectedModel} model)`
          }
        );
      }
    } catch (error) {
      console.error("Prediction error:", error);
      toast.error(
        "Prediction failed",
        {
          id: "prediction",
          description: error instanceof Error ? error.message : "Unknown error"
        }
      );
    } finally {
      setIsPredicting(false);
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
        <div className="max-w-6xl mx-auto space-y-8">
          <div className="text-center space-y-4 animate-slide-in">
            <h1 className="font-display font-bold text-4xl sm:text-5xl">
              Make <span className="gradient-text">Predictions</span>
            </h1>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              Upload new observation data or enter transit parameters manually
            </p>
          </div>

          <ModelSelector
            selectedModel={selectedModel}
            onModelChange={setSelectedModel}
            disabled={isPredicting}
          />

          <div className="grid md:grid-cols-3 gap-6">
            <Card className="p-6 md:col-span-2 bg-card/50 backdrop-blur border-border/40">
              <CardTitle className="flex items-center gap-2">
                <Settings className="w-5 h-5 text-primary" /> Transit Parameters
              </CardTitle>
              <div className="space-y-6">
                <div className="grid md:grid-cols-2 gap-6">
                  {[
                    {
                      id: "period",
                      label: "Orbital Period (days)",
                      placeholder: "3.52",
                      value: manualParams.period,
                    },
                    {
                      id: "duration",
                      label: "Transit Duration (hours)",
                      placeholder: "2.8",
                      value: manualParams.duration,
                    },
                    {
                      id: "depth",
                      label: "Transit Depth (ppm)",
                      placeholder: "0.0012",
                      value: manualParams.depth,
                    },
                    {
                      id: "stellarRadius",
                      label: "Stellar Radius (solar radii)",
                      placeholder: "1.05",
                      value: manualParams.stellarRadius,
                    },
                    {
                      id: "stellarTemp",
                      label: "Stellar Temperature (K)",
                      placeholder: "5778",
                      value: manualParams.stellarTemp,
                    },
                  ].map((field, idx) => (
                    <div
                      key={idx}
                      className={
                        field.id === "stellarTemp" ? "md:col-span-2" : ""
                      }
                    >
                      <label
                        htmlFor={field.id}
                        className="block text-sm font-medium text-muted-foreground mb-1"
                      >
                        {field.label}
                      </label>
                      <input
                        id={field.id}
                        type="number"
                        step={field.id === "depth" ? "0.0001" : "0.01"}
                        placeholder={field.placeholder}
                        value={manualParams[field.id as keyof ManualParams]}
                        onChange={(e) =>
                          setManualParams({
                            ...manualParams,
                            [field.id]: e.target.value,
                          })
                        }
                        className="w-full px-4 py-2 rounded-lg border-2 border-border bg-card/50 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-electric-blue focus:border-electric-blue transition-all duration-300 shadow-sm hover:shadow-electric-blue/30"
                      />
                    </div>
                  ))}
                </div>

                <Button
                  size="lg"
                  className="w-full bg-primary hover:bg-primary/90 text-primary-foreground glow-electric"
                  onClick={handlePredict}
                  disabled={
                    isPredicting || (!uploadedFile && !manualParams.period)
                  }
                >
                  <Sparkles className="mr-2 w-5 h-5 animate-pulse-glow" />
                  {isPredicting ? "Analyzing..." : "Run Prediction"}
                </Button>
              </div>
            </Card>

            <Card className="p-6 bg-card/50 backdrop-blur border-border/40">
              <h2 className="text-xl font-display font-semibold mb-4">
                Upload CSV File
              </h2>
              <div className="border-2 border-dashed border-border/50 rounded-lg p-8 text-center hover:border-primary/50 transition-colors">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                  id="predict-upload"
                />
                <label htmlFor="predict-upload" className="cursor-pointer">
                  <Upload className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
                  <p className="text-sm font-medium">
                    {uploadedFile
                      ? uploadedFile.name
                      : "Click to upload CSV file"}
                  </p>
                  <p className="text-xs text-muted-foreground mt-2">
                    Maximum file size: 50MB
                  </p>
                </label>
              </div>
            </Card>
          </div>

          {predictions.length > 0 && (
            <Card className="bg-card/50 backdrop-blur border-border/40">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>Prediction Results</CardTitle>
                    <CardDescription>
                      {predictions.length} candidates analyzed
                    </CardDescription>
                  </div>
                  <Button variant="outline" size="sm">
                    <Download className="mr-2 w-4 h-4" />
                    Export Results
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Object ID</TableHead>
                      <TableHead>Classification</TableHead>
                      <TableHead>Confidence</TableHead>
                      <TableHead>Period</TableHead>
                      <TableHead>Radius</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {predictions.map((pred, idx) => (
                      <TableRow key={idx}>
                        <TableCell className="font-medium">{pred.id}</TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              pred.prediction === "Exoplanet"
                                ? "default"
                                : "secondary"
                            }
                            className={
                              pred.prediction === "Exoplanet"
                                ? "bg-primary/20 text-primary border-primary/30"
                                : ""
                            }
                          >
                            {pred.prediction}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          {(pred.confidence * 100).toFixed(1)}%
                        </TableCell>
                        <TableCell>{pred.period}</TableCell>
                        <TableCell>{pred.radius}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default PredictPage;
