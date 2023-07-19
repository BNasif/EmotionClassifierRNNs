# Emotion Classifier and Retrieval-Based Response System

This project involves the development of an emotion classifier and integrating it into a retrieval-based response system to produce emotionally articulate chatbot responses.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Methods](#methods)
  - [Preprocessing](#preprocessing)
  - [Model Architecture](#model-architecture)
- [Retrieval-Based Responses](#retrieval-based-responses)
- [Results](#results)
- [Conclusion](#conclusion)
- [Getting Started](#getting-started)
- [Acknowledgements](#acknowledgements)

## Introduction

Emotion expression and recognition is crucial in human communication. This project aims to enhance chatbot-human interaction by making chatbots emotionally aware. It uses a Recurrent Neural Network (RNN) with Gated Recurrent Units (GRU) for emotion classification, and integrates this into a retrieval-based response system.

## Data

Two primary datasets were used:
1. **Emotions dataset for NLP**: Consists of 19,000 sentences labeled with six emotions (joy, sadness, anger, fear, love, and surprise).
2. **EmpatheticDialogues**: A large-scale empathetic dialogue dataset containing 24,850 one-to-one open-domain conversations with 32 emotion labels.

## Methods

### Preprocessing
- Removal of punctuations and conversion to lowercase.
- Tokenization and conversion to integer sequences.
- Padding of sequences to a consistent length.
- One-hot encoding of emotion labels.

### Model Architecture
- Utilizes a Gated Recurrent Unit (GRU) model.
- Embedding layer using pre-trained GloVE word embeddings.
- GRU layer with 128 units, dropout rate of 30%, and L2 activity regularizer.
- Dense output layer with 6 units and softmax activation function for multi-class classification.

## Retrieval-Based Responses

The EmpatheticDialogues dataset was mapped into six emotions. Upon receiving user input, its emotion is classified, and a response is retrieved from the dataset based on contextual similarity using a bag-of-words model and cosine similarity.

## Results

- Emotion classification model achieved an accuracy, recall, and precision score of 91.6%.
- The retrieval-based system produced contextually appropriate responses demonstrating human-like conversation capabilities.

## Conclusion

The project effectively demonstrates an emotionally aware chatbot by integrating an emotion classifier into a retrieval-based response system. Future enhancements can be made by employing larger datasets, refining the retrieval system, and implementing reinforcement learning methods.



## Acknowledgements

- Emotions dataset for NLP, Praveen
- Empathetic Dialogues, Rashkin
- Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)
