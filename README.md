# cnn-review-sentiment-monitor

<br>

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange.svg)
![Keras](https://img.shields.io/badge/Keras-API-red.svg)
![CNN](https://img.shields.io/badge/Model-CNN-blueviolet.svg)
![GloVe](https://img.shields.io/badge/Embedding-GloVe-yellowgreen.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-purple.svg)
![NumPy](https://img.shields.io/badge/NumPy-Computation-navy.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-yellow.svg)
![Gensim](https://img.shields.io/badge/Gensim-GloVe%20Loader-lightgrey.svg)
![Task](https://img.shields.io/badge/Task-Sentiment%20Analysis-success.svg)
![Dataset](https://img.shields.io/badge/Dataset-Product%20Reviews-purple.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

---

# 🎬 Product Review Sentiment Monitoring Using CNN

This project implements a **Convolutional Neural Network (CNN)** for **binary sentiment classification of product reviews**.

The system classifies each review as **positive** or **negative**. Beyond simple classification, the project is designed with a **monitoring purpose** in mind: when the proportion of negative reviews for a product exceeds a defined threshold, the site administrator should be alerted — enabling proactive responses to customer dissatisfaction.

The project focuses on the complete NLP workflow, starting from a **preprocessed unified CSV dataset** (`reviews.csv`), followed by **stratified data splitting, tokenization, GloVe-based embedding, CNN modeling, training, and evaluation**.

Instead of learning word representations from scratch, the project uses **pre-trained GloVe embeddings** (`glove-wiki-gigaword-100`), which were trained on billions of words. This is essential when working with a relatively small dataset, where training embeddings from scratch would inevitably lead to severe overfitting.

The model uses the **last 400 tokens** of each review as input. Because sentiment is often expressed toward the end of a review — in the conclusion or final verdict — `truncating="pre"` is used to preserve the most informative part of the text.

> **Important:** This project is primarily an educational and experimental NLP project. The trained model achieves a validation accuracy of approximately **80.5%**, which is sufficient for a coarse alerting mechanism but should **not** be interpreted as a production-grade sentiment classifier.

---

## 🎯 Project Objectives

The main objectives of this project are:

* Load a preprocessed unified CSV dataset with `id`, `text`, and `label` columns.
* Split the dataset into **train / validation / test** sets using **stratified sampling**.
* Build a word-level tokenizer on the training data only.
* Convert text into padded integer sequences.
* Load pre-trained **GloVe** word embeddings.
* Build an **embedding matrix** aligned with the tokenizer vocabulary.
* Design and train a **1D CNN** classifier for sentiment detection.
* Evaluate the model on the validation set.
* Save the trained model for reuse.
* Lay the groundwork for a **monitoring layer** that alerts administrators when negative feedback exceeds a threshold.

---

# 📂 Project Structure

```text
.
├── saved_models/
│   └── user_comments.keras             # Saved trained CNN model
│
├── reviews.csv                         # Preprocessed unified dataset
│
├── main.py                             # Main project script
│
├── requirements.txt                    # Python dependencies
│
└── README.md                           # Project documentation
```

> **Note:** The raw `.txt` review files are **not** included in this repository to keep it lightweight. Only the unified `reviews.csv` file is provided, which contains everything needed to reproduce the training pipeline.

---

## 📊 Dataset

The repository ships with a single preprocessed dataset file:

```text
reviews.csv
```

The file contains three columns:

| Column  | Type   | Description                                     |
| ------- | ------ | ----------------------------------------------- |
| `id`    | string | Unique review identifier (e.g., `pos_cv000_29590`) |
| `text`  | string | Cleaned and normalized review text              |
| `label` | int    | `1` for positive, `0` for negative              |

The dataset is **balanced**:

| Label | Meaning  | Count |
| ----- | -------- | ----: |
| 1     | Positive | 1000  |
| 0     | Negative | 1000  |

The rows are shuffled once (with a fixed random seed) before being saved.

### Data Source

The reviews were originally sourced from two folders of raw `.txt` files:

```text
data/pos/      → 1000 positive reviews
data/neg/      → 1000 negative reviews
```

Each raw file contained a single review. The file name served as the unique identifier (e.g., `cv000_29590.txt`).

A helper function (`build_dataset()`) was used **once** during development to:

1. Read every `.txt` file from both folders,
2. Clean the review text,
3. Assign a unique ID and a binary label,
4. Shuffle and export the result as `reviews.csv`.

Since `reviews.csv` is included in the repository, **the raw folders are not required** to train the model. The `build_dataset()` function remains in `main.py` for reference and for anyone who wants to rebuild the CSV from the original raw files.

---

# 🔍 Text Cleaning

Raw review text often contains content that is not useful for sentiment classification, such as:

* HTML tags (`<br />`),
* URLs,
* inconsistent casing,
* redundant whitespace.

The `clean_text()` function performs a **lightweight but effective** normalization pipeline:

| Step                | Operation                                |
| ------------------- | ---------------------------------------- |
| Lowercasing         | `text.lower()`                           |
| HTML removal        | Removes `<...>` tags                     |
| URL removal         | Removes `http://...` and `www....`       |
| Whitespace cleanup  | Collapses multiple spaces into one       |

The cleaning is intentionally minimal:

* **Stopwords are kept** because words such as `not`, `never`, `but`, and `very` are critical for sentiment.
* **Stemming and lemmatization are not applied** because pre-trained embeddings were trained on full word forms.

> **Note:** Since `reviews.csv` is already cleaned, `clean_text()` only runs when rebuilding the dataset from raw files via `build_dataset()`.

---

# 🔄 Data Splitting

The dataset is split into three parts using **stratified sampling** to preserve class balance in each subset:

| Dataset    | Percentage | Samples |
| ---------- | ---------- | ------: |
| Training   | 70%        |    1400 |
| Validation | 10%        |     200 |
| Testing    | 20%        |     400 |

The split is performed using `train_test_split` with:

```python
stratify=dataset["label"]
```

This ensures each subset has an equal proportion of positive and negative reviews.

Conceptually:

```text
Full Dataset (2000)
│
├──── 80% ────────────────────┐
│                             │
│                         Train + Val (1600)
│                             │
│                             ├──── 87.5% ────► Train (1400)
│                             │
│                             └──── 12.5% ────► Val   (200)
│
└──── 20% ──────────────────────────────────────► Test  (400)
```

> **Note:** Stratified splitting is essential for evaluating model performance fairly, especially when working with a balanced dataset.

---

# 🧩 Tokenization & Padding

The CNN requires **fixed-length integer sequences** as input.

The pipeline uses the Keras `Tokenizer`:

```python
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
```

Key design choices:

* The tokenizer is **fitted only on the training set** to avoid **data leakage**.
* `num_words=8000` keeps only the **8000 most frequent words**.
* All other words are mapped to a shared `<OOV>` token.
* An `<OOV>` token is added explicitly so that rare words still receive a learnable representation.

After tokenization, sequences are padded to a fixed length:

```python
pad_sequences(
    sequences,
    padding="post",
    truncating="pre",
    maxlen=400,
)
```

| Parameter      | Value  | Reason                                                  |
| -------------- | ------ | ------------------------------------------------------- |
| `padding`      | post   | Shorter reviews are padded at the end                   |
| `truncating`   | pre    | Longer reviews keep their **last** 400 tokens           |
| `maxlen`       | 400    | Fits the CNN's effective receptive field                |

> **Why `truncating="pre"`?** Sentiment is often expressed in the conclusion of a review. Keeping the **last** part of the text preserves the most informative region.

---

# 🧠 Model Architecture

The classifier is a **1D Convolutional Neural Network** designed for short-text sentiment classification.

The architecture is intentionally lightweight to avoid overfitting on a small dataset:

| Layer                     | Configuration       | Purpose                                     |
| ------------------------- | ------------------- | ------------------------------------------- |
| `Embedding`               | 8000 × 100, frozen  | Maps tokens to pre-trained GloVe vectors    |
| `Dropout`                 | 0.3                 | Regularizes token-level representations     |
| `Conv1D`                  | 32 filters, k=4     | Learns local 4-gram patterns                |
| `Dropout`                 | 0.5                 | Regularizes feature maps                    |
| `GlobalMaxPooling1D`      | –                   | Aggregates the strongest signal per filter  |
| `Dense`                   | 32 units, ReLU      | Non-linear projection                       |
| `Dropout`                 | 0.5                 | Regularizes before the output               |
| `Dense`                   | 1 unit, Sigmoid     | Outputs probability of positive sentiment   |

The architecture is implemented as:

```python
model = Sequential([
    Input(shape=(maxlen,)),
    Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False,
    ),
    Dropout(0.3),
    Conv1D(filters=32, kernel_size=4, activation="relu"),
    Dropout(0.5),
    GlobalMaxPooling1D(),
    Dense(32, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])
```

> **Design note:** `GlobalMaxPooling1D` is used instead of `Flatten`. This dramatically reduces the number of parameters and prevents the model from overfitting on a small dataset.

---

# 🧲 Pre-trained GloVe Embeddings

The embedding layer is initialized with pre-trained **GloVe** vectors (`glove-wiki-gigaword-100`), loaded via `gensim`:

```python
glove = api.load("glove-wiki-gigaword-100")
```

For each word in the tokenizer vocabulary, its GloVe vector is copied into an **embedding matrix**:

```python
embedding_matrix = build_embedding_matrix(
    tokenizer, glove, vocab_size, embedding_dim
)
```

Design decisions:

* The embedding matrix shape is `(vocab_size, 100)`.
* Words not found in GloVe remain as **zero vectors**, but the `<OOV>` token receives its own slot.
* The embedding layer is set to **`trainable=False`**, meaning the pre-trained vectors remain frozen during training.

> **Why freeze embeddings?** With only 1400 training samples, fine-tuning 800,000 embedding parameters would quickly overfit. Frozen pre-trained vectors provide a strong semantic prior and let the CNN focus on learning classification patterns.

---

## ⚙️ Model Configuration

The model is compiled using:

| Parameter      | Value                          |
| -------------- | ------------------------------ |
| Optimizer      | Adam                           |
| Learning Rate  | `0.001` (default)              |
| Loss Function  | Binary Cross-Entropy           |
| Metric         | Accuracy                       |

```python
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)
```

---

# 🌱 Reproducibility

To ensure consistent results across runs, a **global random seed** is set before model construction:

```python
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
```

This controls Python's `random`, NumPy, and TensorFlow random number generators.

> **Note:** Small datasets are sensitive to initialization. Setting a seed is essential for comparing experiments fairly.

---

# 🚂 Training

The model is trained with:

| Parameter               | Value    |
| ----------------------- | -------- |
| Maximum Epochs          | `100`    |
| Batch Size              | `32`     |
| Early Stopping Patience | `15`     |
| Minimum Improvement     | `0.0001` |

### Early Stopping

Early stopping monitors validation loss:

```python
EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True,
    min_delta=0.0001,
)
```

If validation loss does not improve for 15 consecutive epochs, training stops and the best weights are restored.

---

# 💾 Model Saving and Loading

The trained model is automatically saved to:

```text
./saved_models/user_comments.keras
```

The training function checks whether this file already exists. If it does, the existing file is removed and training starts fresh. This guarantees that architectural changes take effect immediately.

> **Note:** The current implementation always retrains from scratch. This is intentional during experimentation, where the model architecture is still evolving.

---

# 📊 Evaluation

The model was trained for **84 epochs** before early stopping triggered. The final results on the validation set were:

| Metric            |   Value |
| ----------------- | ------: |
| **Train Accuracy**| `0.8857` |
| **Train Loss**    | `0.3057` |
| **Val Accuracy**  | `0.8050` |
| **Val Loss**      | `0.4306` |

### Interpretation

* **Train accuracy (88.6%)** is meaningfully higher than validation accuracy (80.5%), but the gap is modest.
* **Validation accuracy (80.5%)** is a reasonable result for a CNN with frozen GloVe embeddings on a small dataset.
* The model is **not overfitting catastrophically** — earlier experiments without GloVe reached 100% train accuracy and only ~79% validation accuracy.
* Introducing **pre-trained embeddings** and **dropout** significantly reduced the train–validation gap.

> **Note:** An earlier configuration briefly reached **85.5%** validation accuracy, but subsequent runs with a fixed seed plateaued around **80.5%**. This indicates the earlier result was likely a lucky initialization rather than a stable improvement.

---

# 📉 Results Visualization

Training can be visualized using `history.history`:

```python
import matplotlib.pyplot as plt

plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.legend()
plt.show()
```

This reveals whether the model is still learning or has plateaued.

---

# 🛠️ Utility Functions

The project is organized into modular sections.

## Preprocessing

```text
clean_text()               # optional; only needed for rebuilding the CSV
build_dataset()            # optional; only needed if raw .txt files are available
train_test_valid_dataset()
prepare_data()
build_tokenizer()
texts_to_padded()
```

## Modeling

```text
set_seed()
build_embedding_matrix()
build_model()
```

## Training

```text
fit_model()
```

This structure keeps the workflow modular and makes individual stages easier to inspect, modify, or replace.

---

# 🚀 Usage

## 1. Install Dependencies

Create a Python environment and install the required libraries:

```bash
pip install tensorflow numpy pandas scikit-learn gensim
```

Or use:

```bash
pip install -r requirements.txt
```

---

## 2. Verify the Dataset

The repository already includes the preprocessed dataset:

```text
reviews.csv
```

No further action is needed. If you want to rebuild it from the original raw `.txt` files, place them inside the project as:

```text
data/pos/       → positive reviews (.txt)
data/neg/       → negative reviews (.txt)
```

…and run `build_dataset()` to regenerate `reviews.csv`.

---

## 3. Run the Project

Execute:

```bash
python main.py
```

The script will:

```text
Load reviews.csv
     ↓
Stratified train/val/test split
     ↓
Fit tokenizer on training data
     ↓
Convert text to padded sequences
     ↓
Load pre-trained GloVe embeddings
     ↓
Build embedding matrix
     ↓
Build CNN model
     ↓
Train with early stopping
     ↓
Save trained model
```

---

# 🔧 Customization

Several important parameters can easily be modified.

### Vocabulary Size

```python
VOCAB_SIZE = 8000
```

Controls how many of the most frequent words are kept.

---

### Maximum Sequence Length

```python
MAXLEN = 400
```

Controls how many tokens per review are fed to the model.

---

### Embedding Dimension

```python
EMBEDDING_DIM = glove.vector_size   # 100 for glove-wiki-gigaword-100
```

Determined automatically from the loaded GloVe model.

---

### CNN Architecture

The model can be modified inside `build_model()`:

```python
Conv1D(filters=32, kernel_size=4, activation="relu")
```

Alternatives worth exploring:

* multiple parallel `Conv1D` layers with different kernel sizes (3, 4, 5),
* a `BiLSTM` layer after the CNN,
* unfreezing the embedding layer with a lower learning rate.

---

### Training Parameters

```python
epochs = 100
batch_size = 32
learning_rate = 0.001
```

along with the Early Stopping parameters.

---

# 🧪 Experiments & Findings

Several configurations were tested during development:

| Configuration                              | Val Accuracy |
| ------------------------------------------ | -----------: |
| CNN from scratch, `Flatten`                |       ~79%   |
| CNN + heavier dropout                      |       ~80%   |
| CNN + GloVe (`trainable=False`)            |    **~80.5%**|
| CNN + GloVe + reduced LR                   |       ~81%   |
| CNN + GloVe, several random seeds          |    80–85%    |

Key observations:

* **Pre-trained GloVe** dramatically reduced overfitting compared to training embeddings from scratch.
* **Replacing `Flatten` with `GlobalMaxPooling1D`** cut the number of trainable parameters by roughly an order of magnitude.
* **Reducing learning rate** sometimes improved stability but did not always improve final accuracy.
* Validation accuracy fluctuates by **2–4%** depending on random seed — meaning isolated "best runs" are not reliable evidence of improvement.

> **Lesson:** On small datasets, always evaluate improvements across multiple seeds before drawing conclusions.

---

# ⚠️ Limitations

### 1. Small Dataset

With only 2000 reviews, the model has limited exposure to linguistic diversity. Rare words, sarcasm, and subtle sentiment are difficult to capture.

---

### 2. Domain Sensitivity

The model was trained on movie-style reviews. Its performance on other product categories (electronics, clothing, food) is not guaranteed.

---

### 3. No Handling of Negation Beyond Embeddings

Words like `not good` are handled only implicitly through GloVe. There is no explicit negation scope detection.

---

### 4. Frozen Embeddings

Because the embedding layer is frozen, domain-specific vocabulary (e.g., product names, brand slang) cannot be learned by the model.

---

### 5. Validation, Not Test, Evaluation

The reported accuracy is on the **validation set**. Final evaluation on the held-out **test set** is still pending.

---

# 🔮 Possible Future Improvements

Several extensions could improve the project and provide more meaningful results:

* Unfreeze the embedding layer and fine-tune it with a lower learning rate.
* Use **multiple parallel Conv1D layers** with different kernel sizes (3, 4, 5).
* Add a **BiLSTM** layer on top of the CNN to capture long-range dependencies.
* Compare against **ParsBERT / BERT** for stronger baselines.
* Introduce **class weights** or **focal loss** if the dataset becomes imbalanced.
* Build a **monitoring layer** that tracks negative-review ratios per product over time.
* Add a **REST API** for on-the-fly prediction.
* Deploy a **dashboard** for administrators to visualize sentiment trends.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome.

If you have ideas for improving the sentiment pipeline, feel free to open an **Issue** or submit a **Pull Request**.

---

## 📜 License

This project is licensed under the **MIT License**.

You are free to use, modify, and distribute this project in accordance with the terms of the license.