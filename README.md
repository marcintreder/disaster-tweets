# Classifying Disaster Tweets using Recurrent Neural Networks

This repository contains the complete project for the Kaggle competition, "[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/overview)". The goal is to build a machine learning model that can accurately determine whether a given tweet is about a real disaster or not.

***

## 1. Project Overview

In the age of real-time information, social media platforms like X have become crucial channels for communication during emergencies. However, the language used is often ambiguous. This project tackles the challenge of using Natural Language Processing (NLP) to programmatically distinguish between tweets reporting on actual disasters and those using similar language in a colloquial or figurative way.

The project walks through the entire machine learning workflow:
* **Exploratory Data Analysis (EDA)** to understand the dataset.
* **Text Preprocessing and Cleaning** to prepare the data for modeling.
* **Model Comparison** of four different RNN architectures.
* **Hyperparameter Tuning** to optimize the best-performing model.
* **Final Evaluation** and generation of a Kaggle submission file.

The final, tuned **Bidirectional LSTM** model achieved a **Validation F1-Score of 0.746** for the disaster class.

***

## 2. Architectural Comparison

The four models were trained with a standard set of hyperparameters. The results clearly showed that the gated architectures (LSTM, GRU) dramatically outperformed the Simple RNN. The Bidirectional LSTM was selected for the next stage of optimization due to its high theoretical potential.

| Model              | Validation F1-Score |
| :----------------- | :------------------ |
| Simple RNN         | 0.4271              |
| LSTM               | 0.7225              |
| GRU                | 0.7212              |
| Bidirectional LSTM | 0.7212              |

!(<PATH_TO_YOUR_IMAGE>/initial_comparison_chart.png>)

## 3. Hyperparameter Tuning

After identifying the Bidirectional LSTM as the most promising architecture, I ran a series of experiments to fine-tune it. This process revealed that a simpler model with strong regularization performed best.

| Embedding Dim | LSTM Units | Dropout Rate | Validation F1-Score |
| :------------ | :--------- | :----------- | :------------------ |
| **64** | **64** | **0.5** | **0.7460** |
| 128           | 100        | 0.5          | 0.7454              |
| 64            | 64         | 0.3          | 0.7434              |
| 128           | 64         | 0.5          | 0.7407              |
| 128           | 64         | 0.3          | 0.7396              |
| 64            | 100        | 0.3          | 0.7390              |
| 64            | 100        | 0.5          | 0.7328              |
| 128           | 100        | 0.3          | 0.7305              |

## 4. Final Model Performance

Finally, I trained the Bidirectional LSTM using the winning hyperparameters for a full 5 epochs.

| Metric             | Precision | Recall | F1-Score | Support |
| :----------------- | :-------- | :----- | :------- | :------ |
| **Non-Disaster (0)** | 0.80      | 0.82   | 0.81     | 874     |
| **Disaster (1)** | 0.76      | 0.73   | 0.74     | 649     |
|                    |           |        |          |         |
| **Accuracy** |           |        | 0.78     | 1523    |
| **Weighted Avg** | 0.78      | 0.78   | 0.78     | 1523    |


## 5. How to Run the Project

To replicate the analysis and results, follow these steps:

### Clone the Repository
First, clone this repository to your local machine using git.

```bash
git clone https://github.com/marcintreder/disaster-tweets.git
cd disaster-tweets
```

### Install all the necessary Python libraries using the requirements.txt file.

```bash
pip install -r requirements.txt
```

### Run jupyter notebook

```bash
jupyter notebook
```
