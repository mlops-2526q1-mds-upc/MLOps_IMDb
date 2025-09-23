---
language: en
datasets:
- stanfordnlp/imdb
metrics:
- accuracy
- f1
model-name: "Logistic Regression Sentiment Classifier (IMDB Baseline)"
library_name: scikit-learn
tags:
- sentiment-analysis
- text-classification
- imdb
- baseline
---

# Model Card: Logistic Regression Sentiment Classifier (IMDB Baseline)

## Model Description
This model is a **baseline sentiment classifier** trained on the [IMDB Movie Reviews dataset](https://huggingface.co/datasets/stanfordnlp/imdb).  
It uses a **TF-IDF vectorizer** to transform text into features and a **Logistic Regression** classifier for binary classification (positive or negative sentiment).  

The model is lightweight, interpretable, and serves as a reproducibility baseline for the IMDB dataset.

## Intended Uses
- **Primary use case**: binary sentiment classification on movie reviews.  
- **Educational purpose**: demonstrate reproducibility and versioning with MLOps tools (Git, DVC, MLflow).  
- **Not intended**: for production deployment in high-stakes domains.

## Training Procedure
- **Training data**: 25,000 reviews (IMDB training split).  
- **Validation/testing**: 25,000 reviews (IMDB test split).  
- **Preprocessing**:
  - Tokenization by whitespace and punctuation.
  - Lowercasing of all text.
  - Vectorization with TF-IDF (max_features=20,000).  
- **Algorithm**: Logistic Regression with default scikit-learn parameters.

## Evaluation
- **Metrics**:
  - Accuracy: ~0.82 (typical for this setup).  
  - F1-score: ~0.82.  
- **Baseline comparison**:
  - Better than random guessing (0.5).  
  - Provides a reference point for more complex models (e.g., BiLSTM, BERT).  

## Limitations
- Only supports **English text**.  
- Binary classification oversimplifies sentiment (no neutral/mixed categories).  
- Cannot handle sarcasm, irony, or nuanced emotional expressions.  
- Performance is lower compared to state-of-the-art Transformer models (e.g., BERT).  

## Ethical Considerations
- Potential bias from IMDB users (reviews may not represent all demographics).  
- Limited interpretability of individual coefficients in high-dimensional TF-IDF space.  
- Should not be used in sensitive contexts (e.g., HR screening, social decision-making).  

## Citation
If you use this model or dataset, please cite:  

- Maas, Andrew L., Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts.  
  "Learning Word Vectors for Sentiment Analysis." *Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies*. 2011.

