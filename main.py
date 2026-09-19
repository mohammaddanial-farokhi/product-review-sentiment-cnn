import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Embedding,
    Conv1D,
    MaxPool1D,
    Dropout,
    GlobalMaxPooling1D,
    concatenate,
)
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping
import gensim.downloader as api
import random
import numpy as np
import tensorflow as tf


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


def build_tokenizer(X_train, vocab_size=8000):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    return tokenizer


def texts_to_padded(texts, tokenizer, maxlen=400):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        padding="post",
        truncating="pre",
        maxlen=maxlen,
    )
    return padded


# ==================================
# 2. Modeling
# ==================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_embedding_matrix(tokenizer, glove, vocab_size, embedding_dim):
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    found = 0
    not_found = 0

    for word, index in tokenizer.word_index.items():
        if index >= vocab_size:
            continue

        if word in glove:
            embedding_matrix[index] = glove[word]
            found += 1
        else:
            not_found += 1

    print(f"Words found in GloVe: {found}")
    print(f"unknow words: {not_found}")
    print(f"Coverage rate: {found / (found + not_found) * 100:.1f}%")

    return embedding_matrix


def build_model(vocab_size=12000, embedding_dim=50, maxlen=400, embedding_matrix=None):
    model = Sequential()

    model.add(Input(shape=(maxlen,)))

    if embedding_matrix is not None:
        model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                weights=[embedding_matrix],
                trainable=False,
            )
        )
    else:
        model.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
            )
        )
    model.add(Dropout(0.3))
    model.add(Conv1D(filters=32, kernel_size=4, activation="relu"))
    model.add(Dropout(0.5))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(32, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model


def fit_model(model, X_train, y_train, X_val, y_val):
    MODEL_PATH = "./saved_models/user_comments.keras"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print("Old model removed.")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        min_delta=0.0001,
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop],
        verbose=1,
    )

    model.save(MODEL_PATH)
    print("Model saved successfully.")

    return model, history


if __name__ == "__main__":
    glove = api.load("glove-wiki-gigaword-100")

    VOCAB_SIZE = 8000
    EMBEDDING_DIM = 50
    MAXLEN = 400

    main_dataset = build_dataset()
    train_dataset, test_dataset, val_dataset = train_test_valid_dataset(main_dataset)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(
        train_dataset, val_dataset, test_dataset
    )

    tokenizer = build_tokenizer(X_train)
    X_train_pad = texts_to_padded(X_train, tokenizer, maxlen=400)
    X_val_pad = texts_to_padded(X_val, tokenizer, maxlen=400)
    X_test_pad = texts_to_padded(X_test, tokenizer, maxlen=400)

    glove = api.load("glove-wiki-gigaword-100")
    EMBEDDING_DIM = glove.vector_size

    set_seed(42)

    embedding_matrix = build_embedding_matrix(tokenizer, glove, VOCAB_SIZE, EMBEDDING_DIM)

    model = build_model(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        maxlen=MAXLEN,
        embedding_matrix=embedding_matrix,
    )
    model, history = fit_model(model, X_train_pad, y_train, X_val_pad, y_val)
