---
language: en
license: "IMDB Terms of Use"
dataset_name: "IMDB Movie Reviews"
annotations_creators:
- crowdsourced
task_categories:
- text-classification
task_ids:
- sentiment-classification
paperswithcode_id: imdb
pretty_name: "IMDB Movie Reviews"
size_categories:
- 10K<n<100K
---

# Dataset Card for IMDB Movie Reviews

## Dataset Summary
The IMDB Movie Reviews dataset contains **50,000 reviews** from the Internet Movie Database, labeled for **binary sentiment classification** (positive or negative).  
The dataset is balanced: 25,000 reviews are positive and 25,000 reviews are negative. It is widely used as a benchmark for text classification tasks.

## Motivation
The dataset was created to support **research in sentiment analysis and natural language processing**.  
It provides a realistic benchmark for models that classify text into sentiment categories, helping researchers and practitioners compare algorithms in a reproducible way.

## Dataset Composition
- **Size**: 50,000 movie reviews.  
  - Training set: 25,000 reviews.  
  - Test set: 25,000 reviews.  
- **Labels**: Binary sentiment → {positive, negative}.  
- **Language**: English.  
- **Domain**: Movie reviews from IMDB.

## Collection Process
- Reviews were collected from the Internet Movie Database (IMDB).  
- Neutral reviews (ratings around 5/10) were excluded to ensure clear polarity.  
- The dataset is balanced between positive and negative classes.

## Preprocessing
- Reviews are provided as plain text.  
- No heavy preprocessing is applied (text may include punctuation, capitalization, HTML tags).  
- Users can apply tokenization, lowercasing, or stop-word removal depending on their pipeline.

## Uses
This dataset is suitable for:
- **Binary sentiment analysis**.  
- **Text classification**.  
- **Transfer learning** benchmarks (e.g., fine-tuning Transformer models like BERT).  

It is not designed for:
- Multi-class sentiment (no neutral/mixed labels).  
- Cross-lingual sentiment (only English).

## Limitations and Biases
- **Language restriction**: only English reviews are included.  
- **Binary simplification**: sentiment is reduced to positive/negative, ignoring neutral or nuanced expressions.  
- **Bias**: reviews may reflect cultural and demographic biases of IMDB users.  
- **Domain-specific**: trained models may not generalize well beyond movie reviews.

## Citation
If you use this dataset, please cite:  

Maas, Andrew L., Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts.  
*"Learning Word Vectors for Sentiment Analysis."* Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, 2011.  

