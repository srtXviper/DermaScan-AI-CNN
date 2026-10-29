from __future__ import annotations
import asyncio, io, os
from functools import lru_cache
from pathlib import Path
from typing import Any
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from cnn_model import SkinCNN

APP_NAME = 'DermaScan AI'
MAX_IMAGE_BYTES = 8 * 1024 * 1024
ALLOWED_FORMATS = {'JPEG', 'PNG', 'WEBP'}
MODEL_PATH = Path(os.getenv('DERMASCAN_CNN_PATH', 'models/skin_cnn.pth'))
IMAGE_SIZE = 224
app = FastAPI(title=APP_NAME, description='Educational CNN image-screening prototype; not for diagnosis.', version='0.3.0-cnn')
app.mount('/static', StaticFiles(directory='static'), name='static')

class SymptomRequest(BaseModel):
    duration: str = Field(default='unknown', max_length=40)
    itching: bool = False
    pain: bool = False
    changing: bool = False
    bleeding_or_open_sore: bool = False
    fever_or_rapid_spread: bool = False
    notes: str = Field(default='', max_length=1000)

@lru_cache(maxsize=1)
def get_skin_model():
    """Load a locally trained CNN checkpoint. Never invent labels if weights are absent."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f'CNN weights are not present at {MODEL_PATH}. Train the model with train_cnn.py and add the checkpoint before enabling inference.'
        )
    try:
        checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    except TypeError:  # compatibility with older PyTorch releases
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    classes = checkpoint.get('classes')
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError('Checkpoint must contain a classes list with at least two labels.')
    model = SkinCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, classes, int(checkpoint.get('image_size', IMAGE_SIZE))

def classify_image(image_bytes: bytes) -> list[dict[str, Any]]:
    model, classes, size = get_skin_model()
    with Image.open(io.BytesIO(image_bytes)) as source:
        image = source.convert('RGB').resize((size, size))
    import numpy as np
    tensor = torch.from_numpy(np.asarray(image, dtype='float32').copy()).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.inference_mode():
        scores = torch.softmax(model(tensor), dim=1)[0]
    count = min(3, len(classes))
    values, indices = torch.topk(scores, k=count)
    return [{'label': classes[int(i)], 'score': round(float(s), 4)} for s, i in zip(values, indices)]

@app.get('/')
def home(): return FileResponse('static/index.html')

@app.get('/health')
def health():
    exists = MODEL_PATH.is_file()
    return {'status':'ok', 'app':APP_NAME, 'cnn_architecture':'SkinCNN', 'model_configured':exists,
            'model_weights_loaded':False, 'model_path':str(MODEL_PATH),
            'message':'Checkpoint file found; it is loaded on first inference.' if exists else 'CNN code is installed, but trained weights are missing. Train and add models/skin_cnn.pth.'}

@app.post('/api/image-check')
async def image_check(image: UploadFile = File(...)):
    if image.content_type not in {'image/jpeg','image/png','image/webp'}:
        raise HTTPException(status_code=415, detail='Please upload a JPG, PNG, or WEBP image.')
    content = await image.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail='Image must be 8 MB or smaller.')
    try:
        with Image.open(io.BytesIO(content)) as im:
            im.verify()
        with Image.open(io.BytesIO(content)) as im:
            fmt, width, height = im.format, *im.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=400, detail='The uploaded file could not be read as an image.')
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(status_code=415, detail='Please upload a JPG, PNG, or WEBP image.')
    if width < 64 or height < 64:
        raise HTTPException(status_code=400, detail='Image resolution is too small; use at least 64 × 64 pixels.')
    try:
        predictions = await asyncio.to_thread(classify_image, content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f'CNN inference failed: {str(exc)[:250]}') from exc
    return {'accepted':True, 'filename':image.filename, 'width':width, 'height':height,
            'model_configured':True, 'model_id':'DermaScan-SkinCNN', 'predictions':predictions,
            'message':'Experimental CNN output generated. These labels are not a diagnosis.',
            'mode':'experimental_cnn_classification',
            'disclaimer':'Educational/research use only. This CNN is not clinically validated. Model scores are not calibrated medical probabilities. Do not use this output to diagnose or choose treatment; consult a qualified healthcare professional.'}

@app.post('/api/symptom-summary')
def symptom_summary(data: SymptomRequest):
    urgent = data.bleeding_or_open_sore or data.fever_or_rapid_spread
    prompt_review = data.changing or data.pain or data.duration in {'over_month','2_4_weeks'}
    if urgent:
        urgency='Seek prompt medical advice'
        explanation='Because you reported a bleeding/open sore or fever/rapid spread, contact a healthcare professional promptly. If symptoms are severe or you feel acutely unwell, seek urgent local medical care.'
    elif prompt_review:
        urgency='Arrange a clinician review'
        explanation='A changing, painful, or persistent skin concern should be assessed by a qualified healthcare professional. This questionnaire cannot identify its cause.'
    else:
        urgency='Monitor and seek advice if concerned'
        explanation='This form cannot determine whether a skin concern is harmless. Arrange a professional assessment if it persists, worsens, changes, or worries you.'
    return {'mode':'rule_based_educational_guidance','diagnosis':None,'disease_prediction':None,
            'urgency_label':urgency,'summary':explanation,
            'symptoms_recorded':{'duration':data.duration,'itching':data.itching,'pain':data.pain,'changing':data.changing,
                'bleeding_or_open_sore':data.bleeding_or_open_sore,'fever_or_rapid_spread':data.fever_or_rapid_spread,
                'notes_provided':bool(data.notes.strip())},
            'disclaimer':'Educational guidance only, not a diagnosis or a substitute for medical care. CNN outputs are experimental and not clinically validated.'}
