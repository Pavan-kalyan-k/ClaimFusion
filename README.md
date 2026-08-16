# Car Damage Assessment API

This project integrates three separate models into a unified production-ready FastAPI application:
1. **Computer Vision Model (YOLOv8)**: Detects damaged vehicle parts.
2. **Deep Learning Model (MobileNetV2)**: Assesses damage severity for each part.
3. **Machine Learning Model (Gradient Boosting)**: Estimates the final insurance claim amount.

## Architecture & Data Flow

```
Raw Image -> YOLO Inference -> Bounding Boxes (Parts)
                                    |
                                    v
                            Crop Detected Regions
                                    |
                                    v
                           MobileNetV2 Inference
                                    |
                                    v
                    Max Severity Aggregation & Part Count
                                    |
                                    v
                           Feature Engineering
                                    |
                                    v
                       Gradient Boosting Inference
                                    |
                                    v
                              JSON Response
```

## Running the Application

### 1. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Place Model Files
Ensure the following files are in the root directory:
- `best (1).pt` (YOLO)
- `best_mobilenetv2.keras` (Keras)
- `gradient_boosting_model.pkl` (ML)

*Note: You can override these paths using environment variables `YOLO_MODEL_PATH`, `KERAS_MODEL_PATH`, and `ML_MODEL_PATH`.*

### 4. Start the Backend (API)
Open your CMD terminal, activate your virtual environment, and run:
```bash
uvicorn app.main:app --reload
```

### 5. Start the Frontend (React UI)
Open a **new, second CMD terminal**, navigate to the `frontend` folder, and run the React development server:
```bash
cd frontend
npm install
npm run dev
```
The futuristic ClaimFusion dashboard will now be available at `http://localhost:5173`.

## API Documentation
Once running, navigate to:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## Testing the API
You can test the `/predict` endpoint via Swagger, or via curl:
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@car_accident.jpg;type=image/jpeg'
```
