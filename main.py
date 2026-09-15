import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Embedding,
    Conv1D,
    MaxPool1D,
    Dropout,
)
from tensorflow.keras.layers import concatenate


# ==================================
# 1. PreProcess
# ==================================
def clean_text(text):
    URL_PATTERN = re.compile(r"http\S+|www\.\S+")
    HTML_PATTERN = re.compile(r"<.*?>")
    MULTI_SPACE_PATTERN = re.compile(r"\s+")

    text = text.lower()
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" ", text)
    text = MULTI_SPACE_PATTERN.sub(" ", text).strip()

    return text


def build_dataset():
    pos_dir = "data/pos"
    neg_dir = "data/neg"
    rows = []

    # positive comments
    for filename in os.listdir(pos_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(pos_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = clean_text(f.read().strip())

        if text == "":
            continue

        pos_id = filename[:-4]

        rows.append(
            {
                "id": f"pos_{pos_id}",
                "text": text,
                "label": 1,
            }
        )

    # negative comments
    for filename in os.listdir(neg_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(neg_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = clean_text(f.read().strip())

        if text == "":
            continue

        neg_id = filename[:-4]

        rows.append(
            {
                "id": f"neg_{neg_id}",
                "text": text,
                "label": 0,
            }
        )

    main_dataset = pd.DataFrame(rows, columns=["id", "text", "label"])
    main_dataset = main_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
    main_dataset.to_csv("reviews.csv", index=False, encoding="utf-8-sig")

    ##check
    # print(main_dataset.head())
    # print(main_dataset["label"].value_counts())
    # print(main_dataset.shape)

    return main_dataset


def train_test_valid_dataset(dataset):
    train_val, test = train_test_split(
        dataset,
        test_size=0.2,
        stratify=dataset["label"],
        random_state=42,
    )

    train, val = train_test_split(
        train_val,
        test_size=0.125,
        stratify=train_val["label"],
        random_state=42,
    )

    return train, test, val


def prepare_data(train_df, val_df, test_df):
    X_train = train_df["text"]
    y_train = train_df["label"]

    X_val = val_df["text"]
    y_val = val_df["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_tokenizer(X_train, vocab_size=12000):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    return tokenizer


def texts_to_padded(texts, tokenizer, maxlen=2678):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, padding="post", truncating="post", maxlen=maxlen)
    return padded


if __name__ == "__main__":
    main_dataset = build_dataset()
    train_dataset, test_dataset, val_dataset = train_test_valid_dataset(main_dataset)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        train_dataset, val_dataset, test_dataset
    )

    tokenizer = build_tokenizer(X_train)
