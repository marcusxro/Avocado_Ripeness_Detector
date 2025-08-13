import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from typing import List, Dict
import os

app = FastAPI()

MODEL_PATH = "model.pt"  
model = YOLO(MODEL_PATH, task='detect')
labels = model.names

@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence_threshold: float = 0.3
) -> List[Dict[str, float or str]]:
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        results = model(frame, verbose=False)
        detections = results[0].boxes

        detected_objects = []
        for detection in detections:
            conf = detection.conf.item()
            if conf > confidence_threshold:
                classidx = int(detection.cls.item())
                detected_objects.append({
                    "class": labels[classidx],
                    "confidence": float(conf),
                    "class_id": classidx
                })

        return JSONResponse(content=detected_objects)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)