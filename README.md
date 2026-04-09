# CNN Sentiment Ensemble for McDonald's Reviews

Two experiments in ensemble learning for binary sentiment classification on the [McDonald's Store Reviews](https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews) dataset.

---

## Experiments

### 1. `CNN-Sentiment-Ensemble` — single model baseline
A single CNN trained for sentiment classification. Reviews rated 5 stars are labelled Positive (1), all others Negative (0).

### 2. `CNN-Sentiment-Averaging-Ensemble` — soft averaging ensemble
Three identical CNNs trained independently. Their output probabilities are averaged before thresholding at 0.5 — a soft voting ensemble that reduces variance compared to a single model.

---

## Model Architecture

```
Embedding(10000, 100, input_length=100)
→ Conv1D(128 filters, kernel=5, activation='relu')
→ GlobalMaxPooling1D()
→ Dropout(0.5)
→ Dense(1, activation='sigmoid')
```

Compiled with `binary_crossentropy` loss and `adam` optimizer.

---

## Data pipeline

- Raw review text tokenized via Keras `Tokenizer` (max 10,000 words)
- Sequences padded to length 100
- 80/20 train-test split
- `EarlyStopping` on `val_loss` with `patience=3` and `restore_best_weights=True`
- Tokenizer saved via `pickle` for reuse

---

## Tech Stack

`Python` · `TensorFlow / Keras` · `scikit-learn` · `Pandas` · `NumPy`

---

## Usage

```bash
pip install tensorflow scikit-learn pandas numpy
# Place McDonald_s_Reviews.csv in the working directory
python cnn_sentiment_ensemble.py
```

---

## Notes

University coursework project exploring ensemble methods for NLP. Both scripts share the same architecture — the difference is single model vs. averaged ensemble of 3 models.
