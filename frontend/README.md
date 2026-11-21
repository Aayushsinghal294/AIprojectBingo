# Bingo React dashboard

The React/Vite dashboard uses the local Flask API for image classification and
reads the saved validation reports for its model comparison charts.

## Run locally

1. Install the Python dependencies from the repository root:
   `python -m pip install -r requirements.txt`
2. Generate the model comparison and per-class validation reports:
   `python src/evaluate_models.py`
3. Start the API in one terminal:
   `python src/api.py`
4. Start the dashboard in another terminal:
   `cd frontend`
   `npm install`
   `npm run dev`

Open the local URL printed by Vite. The API stays bound to `127.0.0.1`.
