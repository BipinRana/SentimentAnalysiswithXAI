# SentimentAnalysiswithXAI

This project focuses on sentiment analysis of e-commerce reviews with a strong emphasis on explainability. The goal is not just to classify reviews as positive or negative, but to understand why the model makes its predictions. I plan to start with a Random Forest classifier and apply SHAP or LIME for interpretability, followed by a more advanced BERT-based model. Currently, I'm scraping real customer reviews from Daraz using Selenium to build a practical and relevant dataset.

Project Overview
Building a sentiment analysis model for e-commerce product reviews.

Focused not only on classification but also on model interpretability using Explainable AI.

Reviews are scraped in real-time from Daraz using Selenium, creating a realistic, noisy dataset.

Phase 1 involves a Random Forest model with SHAP/LIME for explanations.

Phase 2 upgrades to a fine-tuned BERT model with corresponding XAI techniques for deep learning.

🎯 Motivation
Understand how users express sentiment in real-world platforms.

Make ML models more transparent and trustworthy, especially in high-impact areas like product recommendations.

Explore and apply XAI techniques to bridge the gap between predictions and reasoning.

⚙️ Tech Stack
Language: Python

Data Collection: Selenium (for web scraping from Daraz)

Data Handling: Pandas, NumPy

ML Models: Scikit-learn (Random Forest, Logistic Regression), Transformers (Hugging Face - BERT)

Explainability Tools: SHAP, LIME

Visualization: Matplotlib, Seaborn, SHAP plots
