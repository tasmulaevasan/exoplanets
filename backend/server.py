from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import json
import pandas as pd
import numpy as np
import io
import asyncio
import logging
from datetime import datetime
from predict import predict_single, predict_batch
from train import train_model
from config import get_available_models, get_model_config
from logger import setup_logging, get_logs, clear_logs, add_custom_log

# Setup logging
logger = setup_logging()

@asynccontextmanager
async def lifespan(app):  # noqa: ARG001
    # Startup
    logger.info("=" * 60)
    logger.info("🚀 Exoplanet Detection API Started")
    logger.info("=" * 60)
    logger.info("Available endpoints:")
    logger.info("  - GET  /health       Health check")
    logger.info("  - POST /predict/manual   Manual prediction")
    logger.info("  - POST /predict/csv      Batch CSV prediction")
    logger.info("  - POST /train            Train model")
    logger.info("  - GET  /train/status     Training status")
    logger.info("  - GET  /logs             Get server logs")
    logger.info("  - POST /logs/clear       Clear logs")
    logger.info("=" * 60)

    # Check if models exist
    try:
        models = get_available_models()
        for model in models:
            status = "✅" if model['available'] else "❌"
            logger.info(f"{status} {model['name']}: {'Available' if model['available'] else 'Not found'}")
    except Exception as e:
        logger.error(f"Failed to check models: {str(e)}")

    yield

    # Shutdown
    logger.info("🛑 Shutting down Exoplanet Detection API")

app = FastAPI(title="Exoplanet Detection API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
training_state = {
    "is_training": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 0,
    "message": "",
    "metrics": {},
    "model_type": None  
}
class PredictManualRequest(BaseModel):
    period: float
    duration: float
    depth: float
    stellarRadius: float
    stellarTemp: float
class TrainRequest(BaseModel):
    modelType: str = "user"  
    selectedDatasets: List[str] = []
    learningRate: Optional[float] = None
    epochs: Optional[int] = None
    batchSize: Optional[int] = None
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Exoplanet Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict_manual": "POST /predict/manual",
            "predict_csv": "POST /predict/csv",
            "train": "POST /train",
            "train_status": "GET /train/status",
            "metrics": "GET /metrics"
        }
    }
@app.get("/health")
async def health_check():
    logger.debug("Health check requested")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
@app.post("/predict/manual")
async def predict_manual(request: PredictManualRequest, modelType: Optional[str] = None):
    try:
        logger.info(f"📊 Manual prediction request received (model: {modelType or 'default'})")
        logger.debug(f"Parameters: period={request.period}, depth={request.depth}")
        input_features = {
            "period": request.period,
            "duration": request.duration,
            "depth": request.depth,
            "star_radius": request.stellarRadius,
            "star_temp": request.stellarTemp,
            "log_period": float(np.log1p(request.period)),
            "insolation": 0.0,
            "log_insolation": 0.0,
            "radius_star_ratio": 0.0,
            "density_proxy": 0.0,
            "radius": request.stellarRadius,
            "source": "manual"
        }
        result = predict_single(input_features, model_type=modelType)
        logger.info(f"✅ Prediction: {result['prediction']} (confidence: {result['confidence']:.2%})")
        return {
            "success": True,
            "predictions": [{
                "id": "MANUAL-001",
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "period": f"{request.period:.2f} days",
                "radius": f"{request.stellarRadius:.2f} R☉"
            }]
        }
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...), modelType: Optional[str] = None):
    try:
        logger.info(f"📁 CSV prediction request: {file.filename} (model: {modelType or 'default'})")
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        logger.info(f"Processing {len(df)} rows from CSV")
        predictions = predict_batch(df, model_type=modelType)
        logger.info(f"✅ CSV prediction complete: {len(predictions)} results")
        return {
            "success": True,
            "total_count": len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"❌ CSV prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/predict/download")
async def download_predictions(file: UploadFile = File(...), modelType: Optional[str] = None):
    try:
        logger.info(f"📥 Download predictions request: {file.filename} (model: {modelType or 'default'})")
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        logger.info(f"Processing {len(df)} rows for download")
        predictions = predict_batch(df, model_type=modelType)

        # Convert predictions to DataFrame
        result_df = pd.DataFrame(predictions)
        logger.info(f"Generated {len(result_df)} predictions")

        # Create CSV output
        output = io.StringIO()
        result_df.to_csv(output, index=False)
        output.seek(0)

        logger.info("✅ Predictions CSV ready for download")
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"}
        )
    except Exception as e:
        logger.error(f"❌ Download predictions failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/train")
async def train_endpoint(
    modelType: str = Form("user"),  
    selectedDatasets: str = Form("[]"),
    learningRate: Optional[float] = Form(None),
    epochs: Optional[int] = Form(None),
    batchSize: Optional[int] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    global training_state
    if training_state["is_training"]:
        logger.warning("⚠️ Training request rejected: training already in progress")
        raise HTTPException(status_code=409, detail="Training is already in progress")
    try:
        logger.info(f"🎓 Training started: model={modelType}")
        config = get_model_config(modelType)
        hyperparams = config["hyperparameters"]
        final_lr = learningRate if learningRate is not None else hyperparams["learning_rate"]
        final_epochs = epochs if epochs is not None else hyperparams["epochs"]
        final_batch = batchSize if batchSize is not None else hyperparams["batch_size"]
        logger.info(f"Training config: lr={final_lr}, epochs={final_epochs}, batch={final_batch}")
        datasets = json.loads(selectedDatasets) if selectedDatasets else []
        custom_data_path = None
        if file:
            contents = await file.read()
            custom_data_path = f"data/uploaded_{file.filename}"
            with open(custom_data_path, 'wb') as f:
                f.write(contents)
        training_state = {
            "is_training": True,
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": final_epochs,
            "message": f"Training {config['name']}...",
            "metrics": {},
            "model_type": modelType
        }
        print(f"[TRAIN ENDPOINT] Starting {config['name']} training...")
        task = asyncio.create_task(run_training(
            learning_rate=final_lr,
            epochs=final_epochs,
            batch_size=final_batch,
            model_type=modelType,
            datasets=datasets,
            custom_data=custom_data_path,
            config=config
        ))
        print(f"[TRAIN ENDPOINT] Background task created: {task}")
        return {
            "success": True,
            "message": f"Training started: {config['name']}",
            "config": {
                "modelType": modelType,
                "modelName": config["name"],
                "learningRate": final_lr,
                "epochs": final_epochs,
                "batchSize": final_batch,
                "datasets": datasets,
                "hyperparameters": hyperparams
            }
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        training_state["is_training"] = False
        raise HTTPException(status_code=500, detail=str(e))
async def run_training(learning_rate, epochs, batch_size, model_type, datasets, custom_data, config):
    global training_state
    try:
        hyperparams = config["hyperparameters"]
        print(f"[TRAINING] Starting {config['name']} with lr={learning_rate}, epochs={epochs}, batch={batch_size}")
        def progress_callback(epoch, total, metrics):
            if "batch" in metrics:
                batch = metrics.get("batch", 0)
                total_batches = metrics.get("total_batches", 1)
                progress_pct = metrics.get("progress_pct", 0)
                print(f"[TRAINING] Epoch {epoch}/{total} - Batch {batch}/{total_batches} - Loss: {metrics.get('loss', 'N/A'):.4f}")
                training_state["current_epoch"] = epoch
                training_state["total_epochs"] = total
                training_state["progress"] = progress_pct
                training_state["metrics"] = metrics
                training_state["message"] = f"Epoch {epoch}/{total} - Batch {batch}/{total_batches}"
            else:
                print(f"[TRAINING] Epoch {epoch}/{total} - Loss: {metrics.get('loss', 'N/A')}, Acc: {metrics.get('accuracy', 'N/A')}")
                training_state["current_epoch"] = epoch
                training_state["total_epochs"] = total
                training_state["progress"] = int((epoch / total) * 100)
                training_state["metrics"] = metrics
                training_state["message"] = f"Epoch {epoch}/{total}"
        print("[TRAINING] Calling train_model...")
        final_metrics = await asyncio.to_thread(
            train_model,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            datasets=datasets,
            custom_data_path=custom_data,
            progress_callback=progress_callback,
            model_output_path=config["model_file"],
            metrics_output_path=config["metrics_file"],
            model_type=model_type
        )
        print(f"[TRAINING] Training completed! Final metrics: {final_metrics}")
        training_state["is_training"] = False
        training_state["progress"] = 100
        training_state["message"] = "Training completed!"
        training_state["metrics"] = final_metrics
    except Exception as e:
        training_state["is_training"] = False
        training_state["message"] = f"Training failed: {str(e)}"
        training_state["metrics"] = {"error": str(e)}

@app.get("/train/status")
async def get_training_status():
    return {
        "is_training": training_state["is_training"],
        "progress": training_state["progress"],
        "current_epoch": training_state["current_epoch"],
        "total_epochs": training_state["total_epochs"],
        "message": training_state["message"],
        "metrics": training_state["metrics"],
        "model_type": training_state.get("model_type")
    }
@app.get("/models")
async def get_models():
    try:
        logger.debug("Fetching available models")
        models = get_available_models()
        logger.info(f"✅ Returned {len(models)} model configs")
        return {
            "success": True,
            "models": models
        }
    except Exception as e:
        logger.error(f"❌ Failed to get models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/metrics")
async def get_metrics():
    try:
        import os
        metrics_path = "models/metrics.json"
        if not os.path.exists(metrics_path):
            return {"error": "No model trained yet"}
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/model/info")
async def get_model_info():
    try:
        import os
        metrics_path = "models/metrics.json"
        if not os.path.exists(metrics_path):
            return {
                "success": False,
                "model_name": "No model trained yet",
                "timestamp": None
            }
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        model_filename = metrics.get("model_filename", "tabtransformer.pth")
        timestamp = metrics.get("timestamp", None)
        return {
            "success": True,
            "model_name": model_filename,
            "timestamp": timestamp,
            "accuracy": metrics.get("accuracy"),
            "auc": metrics.get("auc"),
            "f1_score": metrics.get("f1_score")
        }
    except Exception as e:
        return {
            "success": False,
            "model_name": "Error loading model info",
            "timestamp": None,
            "error": str(e)
        }

@app.get("/logs")
async def get_server_logs(limit: int = 100, level: Optional[str] = None):
    """Get recent server logs"""
    try:
        logs = get_logs(limit=limit, level=level)
        return {
            "success": True,
            "logs": logs,
            "total": len(logs),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get logs: {str(e)}")
        return {
            "success": False,
            "logs": [],
            "error": str(e)
        }

@app.post("/logs/clear")
async def clear_server_logs():
    """Clear log buffer"""
    try:
        clear_logs()
        logger.warning("🗑️ Log buffer cleared by user request")
        return {"success": True, "message": "Logs cleared"}
    except Exception as e:
        logger.error(f"Failed to clear logs: {str(e)}")
        return {"success": False, "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    logger.info(f"🌐 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")