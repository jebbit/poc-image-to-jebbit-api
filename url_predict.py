from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from ultralyticsplus import YOLO
from fastapi.middleware.cors import CORSMiddleware 
from pydantic import BaseModel
import requests
import os
import tempfile


from PIL import Image
from io import BytesIO
import hashlib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model
model = YOLO('foduucom/web-form-ui-field-detection')

# Set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

class ImageURL(BaseModel):
    url: str

@app.get("/predict")
async def predict(url: str):
    try:
        # Download the image from the URL
        response = requests.get(url)
        response.raise_for_status()

        # Create a temporary file to save the downloaded image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_filename = temp_file.name

        # Write the downloaded image data to the temporary file
        with open(temp_filename, 'wb') as buffer:
            buffer.write(response.content)

        # Perform inference using the temporary file path
        results = model.predict(temp_filename)

        print("boxes:")
        print(results[0].boxes.xyxy)

        # Extract boxes, confidence values, and class values
        boxes = results[0].boxes.xyxy.tolist()  # Using xyxy format for bounding boxes
        confs = results[0].boxes.conf.tolist()
        cls = results[0].boxes.cls.tolist()

        # Extract class confidences
        names = model.model.names
        names = [names[i] for i in cls]

        # Delete the temporary file
        os.remove(temp_filename)

        print(model.model.names)

        # Return the results
        return JSONResponse(content={
            "boxes": boxes,
            "confs": confs,
            "cls": cls,
            "names": names,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def crop_image_from_url(url, tags):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    
    cropped_images = []
    for tag in tags:
        print("tag:" + str(tag.name))
        left = tag.x
        top = tag.y
        right = left + tag.width
        bottom = top + tag.height
        
        cropped_img = img.crop((left, top, right, bottom))
        cropped_images.append(cropped_img)

    return cropped_images


from pydantic import BaseModel
from typing import List

class Tag(BaseModel):
    height: float
    name: str
    type: str
    width: float
    x: float
    y: float

class CropRequest(BaseModel):
    url: str
    tags: List[Tag]

import base64

@app.post("/crop/")
async def crop_and_return_base64(request_body: CropRequest):
    try:
        print(request_body)
        cropped_images = crop_image_from_url(request_body.url, request_body.tags)
        base64_list = []
        for cropped_img in cropped_images:
            buffer = BytesIO()
            cropped_img.save(buffer, format="PNG")
            base64_encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')  # Convert to Base64 and then to string
            base64_list.append(base64_encoded)
        return JSONResponse(content={"base64s": base64_list})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
