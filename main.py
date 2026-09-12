import os
import pandas as pd
from sklearn.model_selection import train_test_split


# ==================================
# 1. PreProcess
# ==================================
def create_dataset():
    pos_dir = "data/pos"
    neg_dir = "data/neg"
    rows = []

    # positive comments
    for filename in os.listdir(pos_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(pos_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().strip()

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
            text = f.read().strip()

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


if __name__ == "__main__":
    main_dataset = create_dataset()
    tarin_datset, test_dataset, validation_dataset = train_test_valid_dataset(
        main_dataset
    )
