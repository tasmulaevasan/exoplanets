"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Badge } from "@/components/ui/badge";
import { Loader2, Cpu, Zap, CheckCircle2, XCircle } from "lucide-react";
import { api, ModelConfig } from "@/lib/api";

interface ModelSelectorProps {
  selectedModel: string;
  onModelChange: (modelType: string) => void;
  disabled?: boolean;
}

export default function ModelSelector({
  selectedModel,
  onModelChange,
  disabled = false,
}: ModelSelectorProps) {
  const [models, setModels] = useState<ModelConfig[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await api.getModels();
        setModels(response.models);
      } catch (error) {
        console.error("Failed to fetch models:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, []);

  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Model Selection</CardTitle>
          <CardDescription>Choose which model to train</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model Selection</CardTitle>
        <CardDescription>
          Choose which model to use for predictions
        </CardDescription>
      </CardHeader>
      <CardContent>
        <RadioGroup
          value={selectedModel}
          onValueChange={onModelChange}
          disabled={disabled}
          className="space-y-4"
        >
          {models.map((model) => (
            <div
              key={model.type}
              className={`relative flex items-start space-x-4 rounded-lg border p-4 transition-all ${
                selectedModel === model.type
                  ? "border-primary bg-primary/5"
                  : "border-border hover:border-primary/50"
              } ${!model.available && "opacity-50 cursor-not-allowed"}`}
            >
              <RadioGroupItem
                value={model.type}
                id={model.type}
                disabled={disabled || !model.available}
                className="mt-1"
              />
              <div className="flex-1 space-y-2">
                <div className="flex items-center justify-between">
                  <Label
                    htmlFor={model.type}
                    className="text-base font-semibold cursor-pointer"
                  >
                    {model.name}
                  </Label>
                  <div className="flex items-center gap-2">
                    {model.available ? (
                      <Badge variant="secondary" className="gap-1">
                        <CheckCircle2 className="h-3 w-3" />
                        Trained
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="gap-1">
                        <XCircle className="h-3 w-3" />
                        Not Available
                      </Badge>
                    )}
                    {model.hyperparameters.device === "cuda" ? (
                      <Badge variant="default" className="gap-1">
                        <Zap className="h-3 w-3" />
                        GPU
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="gap-1">
                        <Cpu className="h-3 w-3" />
                        CPU
                      </Badge>
                    )}
                  </div>
                </div>

                <p className="text-sm text-muted-foreground">
                  {model.description}
                </p>

                <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-muted-foreground mt-2">
                  <div>
                    Epochs:{" "}
                    <span className="text-foreground font-mono">
                      {model.hyperparameters.epochs}
                    </span>
                  </div>
                  <div>
                    Batch Size:{" "}
                    <span className="text-foreground font-mono">
                      {model.hyperparameters.batch_size}
                    </span>
                  </div>
                  <div>
                    Embed Dim:{" "}
                    <span className="text-foreground font-mono">
                      {model.hyperparameters.cat_embed_dim}
                    </span>
                  </div>
                  <div>
                    Model Dim:{" "}
                    <span className="text-foreground font-mono">
                      {model.hyperparameters.d_model}
                    </span>
                  </div>
                  <div>
                    Heads:{" "}
                    <span className="text-foreground font-mono">
                      {model.hyperparameters.n_heads}
                    </span>
                  </div>
                  <div>
                    Layers:{" "}
                    <span className="text-foreground font-mono">
                      {model.hyperparameters.n_layers}
                    </span>
                  </div>
                  <div>
                    MLP Hidden:{" "}
                    <span className="text-foreground font-mono">
                      {model.hyperparameters.mlp_hidden}
                    </span>
                  </div>
                  <div>
                    Learning Rate:{" "}
                    <span className="text-foreground font-mono">
                      {model.hyperparameters.learning_rate}
                    </span>
                  </div>
                </div>

                {model.metrics && (
                  <div className="flex gap-4 text-xs mt-2 pt-2 border-t border-border">
                    <div>
                      Accuracy:{" "}
                      <span className="text-foreground font-semibold">
                        {(model.metrics.accuracy * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div>
                      AUC:{" "}
                      <span className="text-foreground font-semibold">
                        {model.metrics.auc.toFixed(4)}
                      </span>
                    </div>
                    <div>
                      F1:{" "}
                      <span className="text-foreground font-semibold">
                        {model.metrics.f1_score.toFixed(4)}
                      </span>
                    </div>
                  </div>
                )}
                {!model.available && (
                  <div className="text-xs text-destructive mt-2">
                    {model.type === "user"
                      ? "Train your custom model on the Train page to use it"
                      : "This model is not available yet"}
                  </div>
                )}
              </div>
            </div>
          ))}
        </RadioGroup>
      </CardContent>
    </Card>
  );
}
