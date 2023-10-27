from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from ultralyticsplus import YOLO
from io import BytesIO

app = FastAPI()

# Load the model
model = YOLO('foduucom/web-form-ui-field-detection')

# Set model parameters
model.overrides['conf'] = 0.25  # NMS confidence threshold
model.overrides['iou'] = 0.45  # NMS IoU threshold
model.overrides['agnostic_nms'] = False  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

import os
import tempfile

@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        # Create a temporary file to save the uploaded image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_filename = temp_file.name

        # Write the uploaded image data to the temporary file
        with open(temp_filename, 'wb') as buffer:
            buffer.write(await file.read())

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
        #confidences = results[0].probs.tolist()

        # Delete the temporary file
        os.remove(temp_filename)

        print(model.model.names)

        # Return the results
        return JSONResponse(content={
            "boxes": boxes,
            "confs": confs,
            "cls": cls,
            "names": names,
            #"confidences": confidences,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)