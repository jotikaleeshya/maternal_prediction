# Maternal Health Risk Prediction System

A machine learning application that predicts maternal health risk levels based on vital health indicators.

## Features

- Risk prediction using Random Forest classifier
- Real-time health monitoring dashboard
- Historical data tracking
- Personalized health recommendations
- Model performance metrics display

## System Requirements

- Python 3.11 or 3.12
- Modern web browser (Chrome, Firefox, Edge)
- 500MB free disk space

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd maternal_prediction
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

Navigate to the backend directory and run the training script:

```bash
cd BE_AI
python train_maternal_model_improved.py
```

This will generate two files:
- `maternal_model.pkl` (trained model)
- `model_metadata.pkl` (model performance metrics)

Training takes approximately 3-5 minutes depending on your hardware.

## Running the Application

You need to run TWO servers simultaneously.

### Terminal 1: Start Backend API Server

```bash
cd BE_AI
python api.py
```

Server will start at: `http://127.0.0.1:5000`

### Terminal 2: Start Frontend HTTP Server

```bash
cd FE_AI
python -m http.server 8000
```

Server will start at: `http://127.0.0.1:8000`

### Access the Application

Open your browser and navigate to:

```
http://127.0.0.1:8000/Pages/index.html
```

## Important Notes

### Do NOT Open HTML Files Directly

Opening HTML files by double-clicking them (file://) will cause CORS errors. Always access the application through the HTTP server at `http://127.0.0.1:8000`.

### Temperature Units

The application uses Celsius for body temperature. Valid range: 35-42°C.

The training data is automatically converted from Fahrenheit to Celsius during preprocessing.

## Input Requirements

- Age: 10-100 years
- Systolic BP: 60-200 mmHg
- Diastolic BP: 40-140 mmHg
- Blood Sugar: 1.0-30.0 mmol/L
- Body Temperature: 35-42°C
- Heart Rate: 40-200 bpm

## API Endpoints

- `GET /health` - Server health check
- `POST /predict` - Make risk prediction
- `GET /history` - Retrieve prediction history
- `GET /stats` - Get health statistics
- `GET /model-metrics` - Get model performance metrics

## Model Information

- Algorithm: Random Forest Classifier
- Test Accuracy: ~73%
- ROC AUC Score: ~90%
- Features: Age, Systolic BP, Diastolic BP, Blood Sugar, Body Temperature, Heart Rate
- Classes: low risk, mid risk, high risk

## Project Structure

```
maternal_prediction/
├── BE_AI/
│   ├── api.py                              # Flask API server
│   ├── train_maternal_model_improved.py    # Model training script
│   ├── Maternal Health Risk Data Set.csv   # Training dataset
│   ├── maternal_model.pkl                  # Trained model (generated)
│   └── model_metadata.pkl                  # Model metadata (generated)
├── FE_AI/
│   ├── Pages/
│   │   ├── index.html                      # Home page
│   │   ├── input.html                      # Health data input form
│   │   ├── result.html                     # Prediction results
│   │   └── history.html                    # Prediction history
│   ├── CSS/
│   │   └── styles.css                      # Application styles
│   └── JS/
│       └── script.js                       # Frontend logic
├── README.md
├── requirements.txt
└── .gitignore
```

## Troubleshooting

### Server Not Connected Error

Make sure both servers are running:
1. Backend API at port 5000
2. Frontend HTTP server at port 8000
3. Access via `http://127.0.0.1:8000/Pages/index.html`, not file://

### Model Files Not Found

Run the training script to generate the model files:

```bash
cd BE_AI
python train_maternal_model_improved.py
```

### Port Already in Use

If port 5000 or 8000 is already in use, either:
- Stop the existing process
- Change the port in api.py (line 245) or use a different port for the HTTP server

### Python Not Found

Use the full path to your Python executable:

```bash
# Example for Windows
"C:\Users\YourName\AppData\Local\Programs\Python\Python312\python.exe" api.py
```

## Development

### Retraining the Model

To retrain the model with updated data or parameters:

```bash
cd BE_AI
python train_maternal_model_improved.py
```

The API server will automatically reload and use the new model.

### Modifying API Validation

Edit validation ranges in `BE_AI/api.py` lines 51-62.

### Updating Frontend

The Flask server runs in debug mode and will auto-reload when you modify `api.py`. The frontend requires a browser refresh to see changes.

## Dataset

Source: Maternal Health Risk Data Set
- Total samples: 1014
- Features: 6 health indicators
- Classes: 3 risk levels (low, mid, high)

## License

This project is for educational and research purposes.

## Contributors

Add your name here when contributing to the project.
