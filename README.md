# Bingo — Waste Intelligence

Bingo classifies waste images into six categories: cardboard, glass, metal,
paper, plastic, and trash. The React dashboard displays image predictions,
confidence scores, model comparisons, and per-class validation scores.

## Requirements
 
- Python with the project dependencies
- Node.js and npm
- The trained model files in `models/` 
  
## Start the app
 
Run these commands from the repository root. 
  
1. Install Python dependencies (once):

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   If you are not using the repository's `.venv`, replace that executable with
   `python`. 

2. Generate or refresh model comparison reports (run when reports are missing
   or models change):

   ```powershell
   .\.venv\Scripts\python.exe src\evaluate_models.py
   ```

3. Start the API in one terminal:

   ```powershell
   .\.venv\Scripts\python.exe src\api.py
   ```

4. In a second terminal, start the React frontend:

   ```powershell
   cd frontend
   npm install
   npm run dev
   ```

5. Open the local URL printed by Vite, usually `http://localhost:5173`.

Keep both terminals running while using the app. Press `Ctrl+C` in each terminal
to stop the frontend and API.

The API listens only on `127.0.0.1:8000`; Vite proxies `/api` requests to it.
Model metrics are based on the validation split, not the separate test split.
