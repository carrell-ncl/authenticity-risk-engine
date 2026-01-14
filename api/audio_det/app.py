import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from src.audio_det.inference.scorer import AudioSpoofScorer, ScorerConfig

CNN_MODEL = os.environ.get("CNN_MODEL", "models/cnn/audio_cnn_balanced_best.pt")
CALIBRATOR = os.environ.get("CALIBRATOR", "models/calibrators/agg_lr_real_or_fake_new.joblib")
DEVICE = os.environ.get("DEVICE", "cpu")

app = FastAPI(title="Audio Authenticity Risk API", version="0.1.0")

scorer = AudioSpoofScorer(
    cnn_model_path=CNN_MODEL,
    calibrator_path=CALIBRATOR if os.path.exists(CALIBRATOR) else None,
    device=DEVICE,
    config=ScorerConfig(n_segments=6, clip_seconds=4.0, cnn_agg="median", low_thr=0.3, high_thr=0.7),
)

@app.get("/health")
def health():
    return {"ok": True}

# @app.post("/v1/score")
# async def score(file: UploadFile = File(...), threshold: float | None = None):
#     audio_bytes = await file.read()
#     if not audio_bytes:
#         raise HTTPException(status_code=400, detail="Empty upload")

#     try:
#         report = scorer.score_bytes(audio_bytes, filename=file.filename or "audio", threshold=threshold)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Scoring failed: {e}")

#     if not report.get("ok", False):
#         return JSONResponse(status_code=422, content=report)

#     return report

import traceback
from fastapi.responses import JSONResponse

@app.post("/v1/score")
async def score(file: UploadFile = File(...), threshold: float | None = None):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    try:
        report = scorer.score_bytes(audio_bytes, filename=file.filename or "audio", threshold=threshold)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "filename": file.filename,
                "error": f"{type(e).__name__}: {e}",
            },
        )

    if not report.get("ok", False):
        return JSONResponse(status_code=422, content=report)

    return report

