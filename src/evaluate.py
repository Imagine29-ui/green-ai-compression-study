import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

CATEGORIES = ["injured_or_dead_people", "rescue_volunteering_or_donation_effort", "sympathy_and_support",
              "infrastructure_and_utility_damage", "not_humanitarian", "caution_and_advice",
              "displaced_people_and_evacuations", "requests_or_urgent_needs",
              "missing_or_found_people", "other_relevant_information"]

def format_prompt(tweet):
    return f"Classify this disaster tweet into one category: injured_or_dead_people, rescue_volunteering_or_donation_effort, sympathy_and_support, infrastructure_and_utility_damage, not_humanitarian, caution_and_advice, displaced_people_and_evacuations, requests_or_urgent_needs, missing_or_found_people, other_relevant_information.\n\nTweet: {tweet}\nCategory:"

def predict(tweet, model, tokenizer):
    prompt = format_prompt(tweet)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=15, do_sample=False)
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    generated = decoded[len(prompt):].strip().lower()
    for cat in CATEGORIES:
        if cat in generated:
            return cat
    return "unmatched"

def load_model(model_path, adapter_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map={"": 0})
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer

def evaluate(test_df, model, tokenizer, model_name="qwen2.5-3b-qlora-finetuned"):
    test_df["predicted"] = test_df["text_clean"].apply(lambda t: predict(t, model, tokenizer))

    unmatched_rate = (test_df["predicted"] == "unmatched").mean()
    print("Unmatched rate:", unmatched_rate)
    print(classification_report(test_df["class_label"], test_df["predicted"]))

    macro_f1 = f1_score(test_df["class_label"], test_df["predicted"], average="macro")
    weighted_f1 = f1_score(test_df["class_label"], test_df["predicted"], average="weighted")
    print("Macro F1:", macro_f1)
    print("Weighted F1:", weighted_f1)

    os.makedirs("results/plots", exist_ok=True)
    cm = confusion_matrix(test_df["class_label"], test_df["predicted"], labels=CATEGORIES)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=CATEGORIES, yticklabels=CATEGORIES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("results/plots/confusion_matrix_llm.png")

    return {"model": model_name, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "unmatched_rate": unmatched_rate}

if __name__ == "__main__":
    model_path = "path/to/qwen2.5-3b-instruct-official"
    adapter_path = "path/to/qlora-adapter-final"
    test_csv_path = "data/processed/test.csv"

    model, tokenizer = load_model(model_path, adapter_path)
    test_df = pd.read_csv(test_csv_path)
    results = evaluate(test_df, model, tokenizer)
    print(results)
