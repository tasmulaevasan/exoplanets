const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface PredictManualRequest {
  period: number;
  duration: number;
  depth: number;
  stellarRadius: number;
  stellarTemp: number;
}

export interface PredictResponse {
  success: boolean;
  predictions: {
    id: string;
    prediction: string;
    confidence: number;
    period: string;
    radius: string;
  }[];
}

export interface ModelHyperparameters {
  cat_embed_dim: number;
  d_model: number;
  n_heads: number;
  n_layers: number;
  mlp_hidden: number;
  dropout: number;
  batch_size: number;
  epochs: number;
  learning_rate: number;
  pseudo_label_start_epoch: number;
  pseudo_label_threshold: number;
  patience: number;
  device: string;
}

export interface ModelConfig {
  type: string;
  name: string;
  description: string;
  hyperparameters: ModelHyperparameters;
  available: boolean;
  can_train: boolean;
  training_time_estimate: string;
  metrics?: {
    accuracy: number;
    auc: number;
    f1_score: number;
    timestamp: string;
  };
}

export interface ModelsResponse {
  success: boolean;
  models: ModelConfig[];
}

export interface TrainRequest {
  modelType: string;
  selectedDatasets: string[];
  learningRate?: number;
  epochs?: number;
  batchSize?: number;
}

export interface TrainResponse {
  success: boolean;
  message: string;
  config: {
    modelType: string;
    modelName: string;
    learningRate: number;
    epochs: number;
    batchSize: number;
    datasets: string[];
    hyperparameters: ModelHyperparameters;
  };
}

export interface TrainingStatus {
  is_training: boolean;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  message: string;
  model_type?: string;
  metrics: {
    loss?: number;
    auc?: number;
    accuracy?: number;
    f1?: number;
    batch?: number;
    total_batches?: number;
    progress_pct?: number;
  };
}

export interface MetricsResponse {
  success: boolean;
  metrics: {
    accuracy: number;
    auc: number;
    f1_score: number;
    confusion_matrix: number[][];
    num_features: number;
    cat_features: number;
    model_params: number;
    training_samples: number;
    validation_samples: number;
  };
}

export interface ModelInfo {
  success: boolean;
  model_name: string;
  timestamp: string | null;
  accuracy?: number;
  auc?: number;
  f1_score?: number;
  error?: string;
}

class ExoplanetAPI {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async predictManual(
    data: PredictManualRequest,
    modelType?: string
  ): Promise<PredictResponse> {
    const url = new URL(`${this.baseUrl}/predict/manual`);
    if (modelType) {
      url.searchParams.append("modelType", modelType);
    }

    const response = await fetch(url.toString(), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`Prediction failed: ${response.statusText}`);
    }

    return response.json();
  }

  async predictCSV(file: File, modelType?: string): Promise<PredictResponse> {
    const url = new URL(`${this.baseUrl}/predict/csv`);
    if (modelType) {
      url.searchParams.append("modelType", modelType);
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(url.toString(), {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`CSV prediction failed: ${response.statusText}`);
    }

    return response.json();
  }

  async downloadPredictions(file: File, modelType?: string): Promise<Blob> {
    const url = new URL(`${this.baseUrl}/predict/download`);
    if (modelType) {
      url.searchParams.append("modelType", modelType);
    }

    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(url.toString(), {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Download failed: ${response.statusText}`);
    }

    return response.blob();
  }

  async train(config: TrainRequest, file?: File): Promise<TrainResponse> {
    const formData = new FormData();
    formData.append("modelType", config.modelType);
    formData.append(
      "selectedDatasets",
      JSON.stringify(config.selectedDatasets)
    );

    if (config.learningRate !== undefined) {
      formData.append("learningRate", config.learningRate.toString());
    }
    if (config.epochs !== undefined) {
      formData.append("epochs", config.epochs.toString());
    }
    if (config.batchSize !== undefined) {
      formData.append("batchSize", config.batchSize.toString());
    }

    if (file) {
      formData.append("file", file);
    }

    const response = await fetch(`${this.baseUrl}/train`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Training failed");
    }

    return response.json();
  }

  async getModels(): Promise<ModelsResponse> {
    const response = await fetch(`${this.baseUrl}/models`);

    if (!response.ok) {
      throw new Error(`Failed to get models: ${response.statusText}`);
    }

    return response.json();
  }

  async getTrainingStatus(): Promise<TrainingStatus> {
    const response = await fetch(`${this.baseUrl}/train/status`);

    if (!response.ok) {
      throw new Error(`Failed to get training status: ${response.statusText}`);
    }

    return response.json();
  }

  async getMetrics(): Promise<MetricsResponse> {
    const response = await fetch(`${this.baseUrl}/metrics`);

    if (!response.ok) {
      throw new Error(`Failed to get metrics: ${response.statusText}`);
    }

    return response.json();
  }

  async getModelInfo(): Promise<ModelInfo> {
    const response = await fetch(`${this.baseUrl}/model/info`);

    if (!response.ok) {
      throw new Error(`Failed to get model info: ${response.statusText}`);
    }

    return response.json();
  }

  async healthCheck(): Promise<{ status: string }> {
    const response = await fetch(`${this.baseUrl}/health`);

    if (!response.ok) {
      throw new Error("API is not healthy");
    }

    return response.json();
  }
}

export const api = new ExoplanetAPI();

export class TrainingStatusPoller {
  private intervalId: NodeJS.Timeout | null = null;
  private callback: (status: TrainingStatus) => void;

  constructor(callback: (status: TrainingStatus) => void) {
    this.callback = callback;
  }

  start(intervalMs: number = 2000) {
    this.stop();

    this.intervalId = setInterval(async () => {
      try {
        const status = await api.getTrainingStatus();
        this.callback(status);

        if (!status.is_training && status.progress === 100) {
          this.stop();
        }
      } catch (error) {
        console.error("Error polling training status:", error);
      }
    }, intervalMs);
  }

  stop() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }
}
