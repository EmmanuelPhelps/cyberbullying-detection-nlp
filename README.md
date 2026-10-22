# Multiclass Cyberbullying Detection Using Classical NLP

## Project Overview

**Research Title:** Beyond Offensive Words: A Comparative Study of Linguistic Features for Multiclass Cyberbullying Detection Using Classical NLP

This project investigates six-category cyberbullying detection
using classical Natural Language Processing and machine learning.

Three classifiers and three TF-IDF feature representations are
compared using stratified cross-validation.

## Research Questions

RQ1: How accurately can Logistic Regression, Multinomial Naive
Bayes and Linear SVM classify tweets into six categories?

RQ2: Which word-level and character-level linguistic features
are most informative for distinguishing the categories?

RQ3: Which categories are most frequently confused, and what
linguistic similarities contribute to these errors?

## Dataset

Source:
https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification

Original filename: cyberbullying_tweets.csv

Original records: 47,692
Cleaned records: 43,884

Input column: tweet_text
Target column: cyberbullying_type

Categories:
- age
- ethnicity
- gender
- religion
- not_cyberbullying
- other_cyberbullying

## Data Preparation

Preprocessing includes:
- Removing exact duplicate records
- Excluding conflicting annotations
- Normalising URLs and user mentions
- Normalising whitespace
- Removing duplicate cleaned texts
- Preventing train-test text overlap
- Stratified 80/20 train-test split

Training records: 35,107
Testing records: 8,777
Random state: 42

## Feature Representations

1. Word TF-IDF: unigrams and bigrams
2. Character TF-IDF: character n-grams (3–5)
3. Combined word and character TF-IDF

## Machine Learning Algorithms

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine

Nine model configurations were compared using
stratified three-fold cross-validation.

The primary selection metric was macro-F1.

## Best Model

Combined TF-IDF + Logistic Regression

Cross-validation accuracy: 87.83%
Cross-validation macro-F1: 86.54%

## Final Test Performance

Accuracy: 88.56%
Macro-precision: 87.42%
Macro-recall: 87.42%
Macro-F1: 87.35%

Test records: 8,777
Correct predictions: 7,773
Incorrect predictions: 1,004

## Per-Class F1 Scores

| Category | F1 |
|---|---:|
| age | 0.9705 |
| ethnicity | 0.9874 |
| gender | 0.8862 |
| not_cyberbullying | 0.6924 |
| other_cyberbullying | 0.7437 |
| religion | 0.9612 |

## Classification Error Analysis

The most frequent confusion occurred between:

- not_cyberbullying -> other_cyberbullying: 259 errors
- other_cyberbullying -> not_cyberbullying: 181 errors

These account for 440 of 1,004 errors (43.8%).

The results suggest that overlapping vocabulary and
contextual ambiguity may contribute to classification errors.

## Robustness Analysis

An additional experiment included 3,221 ambiguous
training records.

Original training records: 35,107
Expanded training records: 38,328

| Metric | Original | Expanded |
|---|---:|---:|
| Accuracy | 88.56% | 88.07% |
| Macro-F1 | 87.35% | 86.88% |

Adding ambiguous annotations reduced macro-F1 by
0.47 percentage points on the same test set.

## Required Python Libraries

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib

Install using:

    python -m pip install pandas numpy matplotlib seaborn scikit-learn joblib

## How to Run

1. Install Python and Jupyter support in VS Code.
2. Install the required Python libraries.
3. Place cyberbullying_tweets.csv in the NLP folder.
4. Open NOTEBOOK.ipynb.
5. Select the appropriate Python environment.
6. Check dataset and output folder paths.
7. Run notebook cells sequentially from Stage 1 to Stage 9.
8. Review the generated files in results/.

## Project Outputs

The results folder contains:
- Model comparison CSV files
- Classification reports
- Final test predictions
- Confusion matrix
- Misclassified tweet records
- Linguistic feature analysis
- Robustness comparison
- Saved best-performing model

## Limitations

- Original annotation reliability requires verification.
- Some categories share similar linguistic patterns.
- TF-IDF has limited contextual understanding.
- The robustness experiment uses one train-test split.
- Results may not generalise to other social media datasets.

## Future Work

Future work could investigate BERT and RoBERTa
using the same six-category classification task.

The aim would be to compare contextual models against
the classical NLP baseline, particularly for ambiguous
cyberbullying categories.

## Authors

Group members:
- [Emmanuel Phelps Afoakwa]
- [Davina Adeboyega]

Institution: National College of Ireland
Module: Text Analytics and Natural Language Processing
Academic Year: 2026/2027
