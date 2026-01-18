# Quick Start Guide - Deepfake Detection System

## Prerequisites
- Python 3.9+
- Trained model checkpoint in `models/image_det/runs_deepfake/`
- (Optional) CUDA for GPU acceleration

## Setup & Running

### 1. Install Dependencies
```bash
# Install API requirements
pip install -r api/requirements.txt

# Or install everything
pip install -r requirements.txt
```

### 2. Start the API
```bash
# From project root
cd c:\Users\Steve\Desktop\dev\authenticity-risk-engine

# Start API server
python -m uvicorn api.image_det.app:app --reload --host 0.0.0.0 --port 8000

# API will be available at:
# - http://localhost:8000
# - API docs: http://localhost:8000/docs
# - Health check: http://localhost:8000/health
```

### 3. Open the Frontend
```bash
# Simply open in browser:
file:///C:/Users/Steve/Desktop/dev/authenticity-risk-engine/frontend/index.html

# Or serve with Python:
cd frontend
python -m http.server 3000
# Then open http://localhost:3000
```

## API Usage Examples

### cURL
```bash
# Single image prediction
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/image.jpg"

# Health check
curl http://localhost:8000/health
```

### Python
```python
import requests

# Single prediction
with open("image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    result = response.json()
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
files = [
    ("files", open("image1.jpg", "rb")),
    ("files", open("image2.jpg", "rb")),
]
response = requests.post(
    "http://localhost:8000/predict/batch",
    files=files
)
results = response.json()
```

### JavaScript/Fetch
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/predict', {
    method: 'POST',
    body: formData
});

const result = await response.json();
console.log(`Prediction: ${result.prediction}`);
console.log(`Confidence: ${result.confidence}`);
```

## Training for Better Generalization

### Download Multiple Datasets
```bash
# FaceForensics++
# https://github.com/ondyari/FaceForensics

# Celeb-DF v2
# https://github.com/yuezunli/celeb-deepfakeforensics

# DFDC
# https://ai.facebook.com/datasets/dfdc/
```

### Combine Datasets
```python
# Organize like this:
data/image_det/combined/
    fake/
        celeb_df_fake_001.jpg
        ff_fake_001.jpg
        dfdc_fake_001.jpg
    real/
        celeb_df_real_001.jpg
        ff_real_001.jpg
        dfdc_real_001.jpg
```

### Train on Combined Data
```bash
python src/image_det/train/train_deepfake.py \
    --data_dir "data/image_det/combined" \
    --out_dir "models/image_det/runs_deepfake" \
    --epochs 25 \
    --batch_size 32 \
    --model_name efficientnet_b3 \
    --lr 1e-4 \
    --patience 5
```

## Testing Generalization

### Test on Multiple Datasets
```python
from src.image_det.inference.evaluate_test_set import evaluate_test_set

# Test on Celeb-DF
metrics1, df1 = evaluate_test_set(
    test_dir=r"data/image_det/Celeb-DF Preprocessed/test",
    sample_per_class=1000
)

# Test on FaceForensics++
metrics2, df2 = evaluate_test_set(
    test_dir=r"data/image_det/FaceForensics++/test",
    sample_per_class=1000
)

# Compare results
print(f"Celeb-DF Accuracy: {metrics1['accuracy']:.2%}")
print(f"FF++ Accuracy: {metrics2['accuracy']:.2%}")
```

## Deployment Options

### Option 1: Docker (Recommended)
```bash
# Build image
docker build -t deepfake-detection .

# Run container
docker run -p 8000:8000 deepfake-detection

# With GPU
docker run --gpus all -p 8000:8000 deepfake-detection
```

### Option 2: Cloud Deployment

**Heroku**
```bash
heroku create deepfake-detection-api
git push heroku main
```

**AWS EC2**
```bash
# SSH into EC2 instance
ssh -i key.pem ubuntu@your-ec2-ip

# Install dependencies and run
git clone your-repo
cd authenticity-risk-engine
pip install -r api/requirements.txt
python -m uvicorn api.image_det.app:app --host 0.0.0.0 --port 8000
```

**Google Cloud Run**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/deepfake-detection
gcloud run deploy --image gcr.io/PROJECT_ID/deepfake-detection --platform managed
```

### Option 3: Production Setup with Nginx
```nginx
# /etc/nginx/sites-available/deepfake-api
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring & Maintenance

### Logging
API logs are automatically generated. View with:
```bash
tail -f api_logs.log
```

### Performance Metrics
Monitor:
- Request latency
- Throughput (requests/second)
- Model accuracy on new data
- Error rates

### Model Updates
1. Train new model version
2. Test extensively
3. Update checkpoint path in API
4. Restart service
5. A/B test if possible

## Troubleshooting

### API won't start
- Check if model checkpoint exists
- Verify Python version (3.9+)
- Check port 8000 is available
- Review logs for errors

### Slow inference
- Use GPU if available
- Reduce image size
- Enable model quantization
- Use ONNX runtime

### Poor accuracy
- Train on more diverse data
- Increase model size (efficientnet_b3/b4)
- Add more augmentation
- Collect difficult examples
- Retrain with more epochs

### CORS errors
- Update `allow_origins` in app.py
- Use proper domain names
- Check API URL in frontend

## Next Steps

1. **Improve Model**
   - Train on combined datasets
   - Test on multiple benchmarks
   - Add ensemble predictions

2. **Enhance API**
   - Add authentication
   - Implement rate limiting
   - Add caching
   - Create API keys

3. **Better Frontend**
   - Build React/Vue app
   - Add batch upload UI
   - Show attention maps
   - Add history/analytics

4. **Production**
   - Set up CI/CD
   - Add monitoring
   - Implement logging
   - Create backup strategies
