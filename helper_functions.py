import numpy as np
import pandas as pd
LABEL_NAMES = {0: "negative", 1: "positive"}

def balanced_subset(dataset, total_size, seed):
    """Return a shuffled subset with the same number of examples per class."""
    labels = np.asarray(dataset["label"])
    classes = np.unique(labels)

    if total_size % len(classes) != 0:
        raise ValueError("total_size must be divisible by the number of classes")

    examples_per_class = total_size // len(classes)
    rng = np.random.default_rng(seed)
    selected_indices = []

    for label in classes:
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < examples_per_class:
            raise ValueError(f"Not enough examples for label {label}")
        rng.shuffle(candidates)
        selected_indices.extend(candidates[:examples_per_class])

    selected_indices = np.asarray(selected_indices)
    rng.shuffle(selected_indices)
    return dataset.select(selected_indices.tolist())


def label_distribution(dataset):
    counts = pd.Series(dataset["label"]).value_counts().sort_index()
    return {LABEL_NAMES[int(label)]: int(count) for label, count in counts.items()}

def get_label_distribution(train_reviews, validation_reviews, test_reviews):
    distribution_table = pd.DataFrame(
        {
            "Training": label_distribution(train_reviews),
            "Validation": label_distribution(validation_reviews),
            "Test": label_distribution(test_reviews),
        }
    ).T

    distribution_table.index.name = "Split"
    return distribution_table

def get_demo_tokenization_table(tokenizer):
    demo_review = "The performances were excellent, but the story was painfully predictable."

    demo_encoding = tokenizer(
        demo_review,
        padding="max_length",
        truncation=True,
        max_length=20,
    )

    demo_table = pd.DataFrame(
        {
            "position": range(len(demo_encoding["input_ids"])),
            "token": tokenizer.convert_ids_to_tokens(demo_encoding["input_ids"]),
            "input_id": demo_encoding["input_ids"],
            "attention_mask": demo_encoding["attention_mask"],
        }
    )

    print("Original review:")
    print(demo_review,end="\n\n")
    return demo_table