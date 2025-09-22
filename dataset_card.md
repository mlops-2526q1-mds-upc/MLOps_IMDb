# Dataset Card for IMDB Movie Reviews

## Dataset Summary
The IMDB dataset contains 50,000 movie reviews from the Internet Movie Database (IMDB), labeled for sentiment analysis.  
The dataset is balanced, with 25,000 positive and 25,000 negative reviews. It is commonly used for binary text classification tasks.

## Motivation
The dataset was created to support research in sentiment classification and natural language processing.  
Its goal is to evaluate models’ ability to detect sentiment polarity in natural text.

## Dataset Composition
- **Size**: 50,000 reviews (25k for training, 25k for testing).  
- **Labels**: Binary (positive / negative).  
- **Data type**: Text reviews in English.

## Collection Process
The reviews were collected from the Internet Movie Database (IMDB). Reviews with neutral ratings were not included.

## Preprocessing
- Reviews are stored as plain text.  
- Labels are provided separately.  
- The dataset is already split into training and test sets.

## Usage
The IMDB dataset can be used for:
- Sentiment analysis
- Text classification
- Transfer learning for NLP

## Limitations and Biases
- The dataset is only in English.  
- Binary labels oversimplify human sentiment (no neutral or mixed reviews).  
- May contain cultural or linguistic bias (reviews from IMDB users may not represent all audiences).
