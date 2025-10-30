# Customer Reviews Keyword Extraction (Negative Reviews)

**Purpose:** Demo project that generates synthetic customer reviews (because no real dataset was provided) and demonstrates keyword extraction focussed on **negative** reviews using TF-IDF.

**Files in this repo**
- `customer_reviews_keyword_extraction.ipynb` — runnable Jupyter / Colab notebook (main demo).
- `extract_keywords.py` — small script to compute top negative keywords from a CSV.
- `requirements.txt` — Python packages used.
- `.gitignore` — common ignores.
- `synthetic_reviews.csv` — a generated synthetic dataset for quick testing.

**Important:** The dataset in the notebook is *synthetically generated* for demonstration only.

## How to run

### Option A — Open in Google Colab
1. Upload `customer_reviews_keyword_extraction.ipynb` to GitHub (or directly open in Colab via `File -> Open notebook -> GitHub`).
2. In Colab, run the cells. The notebook installs `nltk`, downloads stopwords, and creates a synthetic dataset.

### Option B — Run script locally
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. If you have a CSV `reviews.csv` with columns `review` and `sentiment` (values `negative` or `positive`), run:
```bash
python extract_keywords.py --input reviews.csv --output negative_keywords.csv
```

## Reproducibility & Notes
- The synthetic data generator is seeded for reproducibility.
- Keyword extraction uses TF-IDF with 1-2 grams and a simple scoring (document frequency × average TF-IDF).
- For production or larger datasets, consider more advanced methods (YAKE, RAKE, KeyBERT, embedding + clustering, or domain-specific preprocessing).

## License
This is example/demo code — use freely for learning.
