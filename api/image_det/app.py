"""
FastAPI Application for Deepfake Image Detection

This API provides endpoints for:
- Single image prediction
- Batch image prediction
- Model health checks
- Prediction explanations (optional)

Usage:
    uvicorn api.image_det.app:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import logging
from pathlib import Path
import time
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="API for detecting deepfake/manipulated images using deep learning",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Response Models
class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    filename: str
    prediction: str = Field(..., description="Predicted class: 'real' or 'fake'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    probability_real: float = Field(..., ge=0.0, le=1.0)
    probability_fake: float = Field(..., ge=0.0, le=1.0)
    inference_time_ms: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    predictions: List[PredictionResponse]
    total_images: int
    total_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str
    model_name: str


# Global model instance
class ModelService:
    """Singleton service for model management"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.transform = None
        self.model_path = None
        self.model_name = None
        self.loaded = False
    
    def load_model(
        self,
        checkpoint_path: str = "models/image_det/runs_deepfake/20260116_171601/best.pt",
        device: Optional[str] = None
    ) -> None:
        """Load model from checkpoint"""
        
        if self.loaded:
            logger.info("Model already loaded")
            return
        
        try:
            # Set device
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            
            logger.info(f"Loading model from: {checkpoint_path}")
            logger.info(f"Using device: {self.device}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            cfg = checkpoint.get("cfg", {})
            self.model_name = cfg.get("model_name", "efficientnet_b0")
            img_size = cfg.get("img_size", 224)
            
            # Build model architecture
            self.model = self._build_model(self.model_name)
            self.model.load_state_dict(checkpoint["model"])
            self.model.to(self.device)
            self.model.eval()
            
            # Setup transforms
            self.transform = transforms.Compose([
                transforms.Resize(int(img_size * 1.15)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
            self.model_path = checkpoint_path
            self.loaded = True
            
            logger.info(f"Model loaded successfully: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _build_model(self, model_name: str) -> nn.Module:
        """Build model architecture"""
        if model_name == "efficientnet_b0":
            m = models.efficientnet_b0(weights=None)
            in_features = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(in_features, 1)
            return m
        
        if model_name == "efficientnet_b3":
            m = models.efficientnet_b3(weights=None)
            in_features = m.classifier[1].in_features
            m.classifier[1] = nn.Linear(in_features, 1)
            return m
        
        if model_name == "resnet50":
            m = models.resnet50(weights=None)
            in_features = m.fc.in_features
            m.fc = nn.Linear(in_features, 1)
            return m
        
        raise ValueError(f"Unsupported model: {model_name}")
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Run prediction on a single image"""
        
        if not self.loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            logits = self.model(input_tensor)
            prob_real = torch.sigmoid(logits.squeeze()).item()
            prob_fake = 1 - prob_real
            
            prediction = "real" if prob_real >= 0.5 else "fake"
            confidence = max(prob_real, prob_fake)
            
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "probability_real": prob_real,
                "probability_fake": prob_fake,
                "inference_time_ms": inference_time
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise


# Initialize model service
model_service = ModelService()


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        # Try to find the most recent checkpoint
        import glob
        checkpoints = glob.glob("models/image_det/runs_deepfake/*/best.pt")
        
        if checkpoints:
            # Sort by modification time, get most recent
            checkpoints.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
            checkpoint_path = checkpoints[0]
            logger.info(f"Found checkpoint: {checkpoint_path}")
        else:
            # Fallback to default path
            checkpoint_path = "models/image_det/runs_deepfake/20260116_171601/best.pt"
            logger.warning(f"Using default checkpoint path: {checkpoint_path}")
        
        model_service.load_model(checkpoint_path)
        logger.info("API ready for requests")
        
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        logger.warning("API started but model not loaded. Predictions will fail.")


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Deepfake Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_service.loaded else "model_not_loaded",
        model_loaded=model_service.loaded,
        device=str(model_service.device) if model_service.device else "unknown",
        model_name=model_service.model_name or "unknown"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict if an uploaded image is real or fake
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        Prediction result with confidence scores
    """
    
    if not model_service.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run prediction
        result = model_service.predict(image)
        
        return PredictionResponse(
            filename=file.filename,
            prediction=result["prediction"],
            confidence=result["confidence"],
            probability_real=result["probability_real"],
            probability_fake=result["probability_fake"],
            inference_time_ms=result["inference_time_ms"],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict multiple images in batch
    
    Args:
        files: List of image files
    
    Returns:
        Batch prediction results
    """
    
    if not model_service.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 images per batch")
    
    start_time = time.time()
    predictions = []
    
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                logger.warning(f"Skipping non-image file: {file.filename}")
                continue
            
            # Read and predict
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            result = model_service.predict(image)
            
            predictions.append(PredictionResponse(
                filename=file.filename,
                prediction=result["prediction"],
                confidence=result["confidence"],
                probability_real=result["probability_real"],
                probability_fake=result["probability_fake"],
                inference_time_ms=result["inference_time_ms"],
                timestamp=datetime.now().isoformat()
            ))
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            # Continue with other files
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_images=len(predictions),
        total_time_ms=total_time
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
