# Sentiment Classification with Model Interpretability

An explainable sentiment analysis system for Nepali-English code-mixed text, built on real-world reviews scraped from Daraz Nepal and YouTube. The project goes beyond classification accuracy — it uses SHAP and LIME to make model predictions transparent and interpretable.

---

## 🗂️ Project Overview

- Collected and curated a dataset of **190,000+ user-generated reviews** from Daraz Nepal (custom Selenium scraper), YouTube (YouTube Data API), and Twitter
- Manually labeled a substantial portion of the dataset to establish high-quality ground truth for supervised learning
- Built a comprehensive **preprocessing pipeline** specifically designed for noisy, code-mixed Romanized Nepali-English text
- Trained and evaluated **Logistic Regression, Linear SVC, and Random Forest** classifiers using TF-IDF features
- Experimented with **BERT as a feature extractor** — it underperformed TF-IDF, confirming that well-engineered domain-specific features can outperform generic pretrained embeddings on low-resource code-mixed languages
- Integrated **SHAP and LIME** for word-level explainability, exposing model biases and sentiment-driving features
- Deployed as a **full-stack web application** (Flask + Next.js) with real-time predictions and interactive explanation visualizations

---

## 🎯 Motivation

User-generated content on Nepali platforms like Daraz and YouTube is written in a unique mix of English, Nepali (Devanagari), and Romanized Nepali — a low-resource, non-standardized linguistic space with almost no existing labeled datasets. Standard NLP tools and pretrained models fail here.

Beyond the language challenge, most sentiment classifiers are black boxes. This project treats **explainability as a first-class requirement**: not just predicting sentiment, but showing *why* the model made a prediction, and revealing where it goes wrong.

---

## ⚙️ Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Collection | Selenium WebDriver, YouTube Data API |
| Data Handling | Pandas, NumPy |
| Preprocessing | NLTK, spaCy, regex, emoji (demojize) |
| ML Models | Scikit-learn (Logistic Regression, Linear SVC, Random Forest) |
| Deep Learning | PyTorch, Transformers (BERT — feature extraction) |
| Explainability | SHAP, LIME |
| Visualization | Matplotlib, Seaborn, Plotly, SHAP plots |
| Backend | Flask |
| Frontend | Next.js |

---

## 🔧 Preprocessing Pipeline

Handling code-mixed, informal Nepali text required a custom pipeline far beyond standard NLP preprocessing:

- **Unicode casefolding** — handles non-Latin scripts more robustly than simple lowercasing
- **Noise removal** — URLs, @mentions, hashtag symbols (text preserved)
- **Character & word repetition normalization** — `"sooooo goood"` → `"soo good"` (preserves emphasis signal)
- **Entity replacement** — person names, company names, and product names replaced with semantic tags (`<NAME>`, `<COMPANY>`, `<PRODUCT>`)
- **Number, date, and time tagging** — replaced with `<NUMBER>`, `<DATE>`, `<TIME>` tokens
- **Emoji conversion** — emojis converted to text descriptions via `demojize` (e.g., 😍 → `smiling_face_with_heart_eyes`) rather than removed, preserving strong sentiment signals
- **Romanized Nepali spelling normalization** — manually curated dictionary mapping spelling variants to canonical forms (e.g., `"adkyo"`, `"adkiyo"`, `"adkeyo"` → `"adkyo"`)
- **Stopword retention** — intentionally kept; words like `"not"`, `"never"`, `"very"` are critical for negation and intensity

---

## 📊 Model Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | **93.0%** | **89.2%** |
| Linear SVC | 92.0% | 88.0% |
| Random Forest | 90.0% | 84.5% |

Logistic Regression with TF-IDF features achieved the best performance. BERT-based feature extraction (without fine-tuning) significantly underperformed — standard BERT has minimal exposure to Romanized Nepali vocabulary, producing unreliable embeddings for code-mixed text.

---

## 🔍 Explainability

SHAP and LIME provide complementary interpretability:

- **SHAP** — game-theoretic Shapley values assign each word a contribution score; offers both local (per-prediction) and global (dataset-wide) insights
- **LIME** — approximates the model locally with an interpretable surrogate; intuitive instance-level explanations

Explainability analysis also revealed real model weaknesses:
- **Sentiment incongruence** — a single heart emoji could override multiple negative words, causing a negative review to be classified as positive
- **Biased neutral words** — the Nepali plural marker `"haru"` was consistently misclassified as negative due to spurious training data correlations

These are not just bugs — they are documented limitations that motivate future work with context-aware models like mBERT.

---

## 🚀 Web Application

The full system is deployed as a web app:
- Enter any Nepali-English review and get an instant sentiment prediction
- View confidence scores across positive, neutral, and negative classes
- Explore word-level SHAP and LIME explanation charts showing which words drove the prediction

**Stack:** Flask (backend API) + Next.js (frontend)

---

## 🔮 Future Enhancements

- **mBERT continued pre-training + fine-tuning** — adapt multilingual BERT on in-domain code-mixed data, then fine-tune on the labeled dataset for context-aware sentiment classification
- **Robust class balancing** — stratified sampling, synthetic augmentation, and class-weighted loss to improve minority-class recall
- **Addressing TF-IDF bias** — replace bag-of-words representations with sequence-aware contextual embeddings to eliminate sentiment incongruence issues

---

## 👥 Team

Developed as a final year project at **Cosmos College of Management & Technology** (affiliated to Pokhara University).

- Ankit Nepali
- [Bardan Babu Shiwakoti](https://github.com/aadit1011)
- Bipin Rana
- [Roshan Pandey](https://github.com/roshanpandey-1)

Supervised by **RobinHood Khadka**, Department of ICT.
