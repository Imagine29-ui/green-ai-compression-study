import re
import pandas as pd
from sklearn.model_selection import train_test_split

MAX_LENGTH = 128  # covers 99th percentile of token lengths (86), with margin

def clean_tweet(text):
    text = text.encode('utf-8', 'ignore').decode('utf-8')
    text = re.sub(r'^RT\s+@\w+:\s*', '', text)
    return text.strip()

def load_and_split(df, test_size=0.3, val_size=0.5, seed=42):
    df = df.copy()
    df["text_clean"] = df["tweet_text"].apply(clean_tweet)
    df = df.drop_duplicates(subset="text_clean")
    train_df, temp_df = train_test_split(df, test_size=test_size, stratify=df["class_label"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=val_size, stratify=temp_df["class_label"], random_state=seed)
    return train_df, val_df, test_df
