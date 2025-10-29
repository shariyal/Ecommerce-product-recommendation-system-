# Ecommerce Product Recommendation

This repository contains a small prototype recommendation system using a combination of content-based and collaborative approaches, a small Flask UI/API, and optional LLM-backed explanation helpers.

Contents
- `recommendation_code.py` — core recommendation utilities (data loader, CB/CF/hybrid recommenders, persistence, LLM explainer wrapper).
- `app.py` — small Flask app exposing a UI and REST endpoints for recommendations and interactions.
- `clean_data.csv` — dataset (committed) used for demos and training.
- `trending_products.csv` — optional dataset used for previews.
- `.gitignore` — project ignore rules (virtualenvs, DBs, logs, etc.).

Quick summary
- CLI: run the main script to validate data, generate sample recommendations and persist product metadata to `recommender.db`.
- Web UI: `app.py` serves a small demo UI at `/main` and provides JSON API endpoints under `/api/v1/`.

Prerequisites
- Windows (tested in PowerShell)
- Python 3.10+ recommended
- Git (optional)

Set up (recommended)
Open PowerShell in the repository root and run:

```powershell
# create and activate virtual environment (one-time)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# upgrade pip and install dependencies
pip install --upgrade pip setuptools wheel
pip install -r .\requirements.txt
```

Run the CLI (quick validation)

```powershell
# activate env (if not already)
.\.venv\Scripts\Activate.ps1
python .\recommendation_code.py
```

You should see dataset validation output and example recommendations printed; product metadata will be persisted to `recommender.db`.

Run the Flask UI / API

```powershell
.\.venv\Scripts\Activate.ps1
python .\app.py
```

Open http://127.0.0.1:5000/main to use the simple HTML UI. The UI form posts to `/recommendations` and displays content-based recommendations.

API examples (PowerShell)

Content-based recommendations (POST)
```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/v1/recommendations/content-based' -Method POST -Body (ConvertTo-Json @{product_name='OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath'; top_n=5}) -ContentType 'application/json'
```

Collaborative recommendations (POST)
```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/v1/recommendations/collaborative' -Method POST -Body (ConvertTo-Json @{user_id=1705736792; top_n=5}) -ContentType 'application/json'
```

Hybrid recommendations (POST)
```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/v1/recommendations/hybrid' -Method POST -Body (ConvertTo-Json @{user_id=1705736792; product_name='OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath'; top_n=5}) -ContentType 'application/json'
```

Search products (GET)
```powershell
Invoke-RestMethod -Uri 'http://127.0.0.1:5000/api/v1/products/search?q=bubble' -Method GET
```

Notes on product matching and Tags
- The server will attempt case-insensitive fuzzy matching for product names. If the DB-backed products table doesn't contain the requested Name, the app falls back to the full `clean_data.csv` dataset to find matches.
- If your database table lacks a `Tags` column, the Flask app will synthesize one at runtime from `Category`, `Brand`, `Name`, and `Description` so content-based TF-IDF still works.

LLM explanations (optional)
- The project includes a `RecommendationExplainer` wrapper that supports multiple providers (Gemini/OpenAI/Ollama) and an SQLite cache. Explanations are optional and only used if provider credentials are present.
- Environment variables the explainer may read:
  - `GOOGLE_API_KEY` or `GEMINI_API_KEY` for Google Gemini
  - `OPENAI_API_KEY` for OpenAI
  - `OLLAMA_URL`/`OLLAMA_KEY` for local Ollama (if configured)

Troubleshooting
- numpy/pandas binary mismatch error ("numpy.dtype size changed...") — usually caused by using the wrong Python interpreter or incompatible wheel versions:
  - Ensure you activated the repository venv: `.\.venv\Scripts\Activate.ps1`
  - Then install compatible packages:

```powershell
pip install --upgrade pip setuptools wheel
pip install -U numpy pandas
# or pin to known-good versions
pip install 'numpy==1.26.*' 'pandas==2.1.*'
```

- If the web UI returns "No recommendations available" for a product you expect to exist:
  - Try one of the sample product names below (exact or close match). The app will also return suggestions in the UI if none are found.

Sample product names to test
- OPI Infinite Shine, Nail Lacquer Nail Polish, Bubble Bath
- OPI Nail Lacquer - Dont Bossa Nova Me Around - NL A65
- OPI Infinite Shine 2 Polish - ISL P33 - Alpaca My Bags
- La Vie Organic Cotton Top Sheet* Panty Liners, Ultra Thin, 108 Count
- Egyptian Magic All Purpose Skin Cream 4 Ounce Jar All Natural Ingredients
- EcoTools Ultimate Shade Duo Eyeshadow Makeup Brush Set, 2pc
- Onyx Professional LED Mirror with In-Base Storage and Magnifying Mirror, White
- Schick Quattro for Women Razor Refill, Ultra Smooth, 8 Ct

Database and persistence
- A small SQLite DB `recommender.db` is created when you run the CLI or persist products; it stores product metadata and an LLM explanations cache.
- `recommender.db` is included in `.gitignore` to avoid accidentally committing it.

Development tips
- Add `--no-reload` if the Flask reloader is getting in the way when debugging complex state:

```powershell
python -m flask run --no-reload
```

- If you plan to commit the repo without the CSV files, uncomment the dataset ignore lines in `.gitignore`.

Contributing
- If you want help wiring the UI suggestions into `main.html`, adding CLI flags, or persisting synthesized Tags back to the DB on startup, tell me which feature you want and I can implement it.

License
- This is prototype code for demonstration and learning. Add a license file if you want to open-source it.

---
Generated on October 13, 2025
