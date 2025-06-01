import json
from functools import partial
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
import re

# Check if we're running Gemma-3 models and disable torch compile if needed
import sys
if __name__ == "__main__":
    # Parse arguments early to check model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=str, default="sample")
    parser.add_argument("--model_idx", type=int, default=None)
    args = parser.parse_args()
    
    # If a specific model is selected, check if it's Gemma-3
    if args.model_idx is not None:
        models_temp = [
            "google/gemma-2-2b-it",
            "google/gemma-2-9b-it",
            "google/gemma-2-27b-it",
            "google/gemma-3-4b-it",
            "google/gemma-3-12b-it",
            "Qwen/Qwen3-8B",
            "Qwen/Qwen3-14B",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
        ]
        if args.model_idx < len(models_temp) and "gemma-3" in models_temp[args.model_idx].lower():
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            print(f"Disabled torch dynamo for {models_temp[args.model_idx]}")
    else:
        # If running all models, disable dynamo as we'll hit Gemma-3 models
        os.environ["TORCHDYNAMO_DISABLE"] = "1"
        print("Disabled torch dynamo for compatibility with Gemma-3 models")

import torch
from torch.utils.data import DataLoader

models = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "google/gemma-3-4b-it",
    "google/gemma-3-12b-it",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
]

def extract_first_number(text: str) -> int:
    # Use regex to find the first number in the text
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    return -1  # Return -1 if no number found

def load_dataset(data_dir: str):
    ds = []
    with open(data_dir, "r") as f:
        for line in f:
            ds.append(json.loads(line))
    return Dataset.from_list(ds)


def add_generation_prompt(prompt, tokenizer, model_thinks=True):
    chat_template_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if model_thinks:
        chat_template_kwargs["enable_thinking"] = False

    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **chat_template_kwargs)


def collate_fn(batch, tokenizer, model_thinks=True, instruct=True):
    if instruct:
        formatted_prompts = [add_generation_prompt(item["text"], tokenizer, model_thinks) for item in batch]
        inputs = tokenizer(formatted_prompts, padding=True, truncation=True, return_tensors="pt", add_special_tokens=False)
    else:
        prompts = [item["text"] for item in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "answer": [item["answer"] for item in batch],
        "target_type": [item["target_type"] for item in batch],
    }





if __name__ == "__main__":
    difficulty = args.difficulty
    assert difficulty in ["easy", "medium", "hard", "sample"]

    ds = load_dataset(f"dataset/dataset_instruct_{difficulty}.jsonl")

    results = {}
    save_dir = f"eval/{difficulty}/results.json"
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Select models to run
    if args.model_idx is not None:
        models_to_run = [models[args.model_idx]]
    else:
        models_to_run = models

    for model_name in models_to_run:
        eval_log_file = f"eval/{difficulty}/log/{model_name}.txt"
        os.makedirs(os.path.dirname(eval_log_file), exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if tokenizer.padding_side == "right":
            print("Setting padding side to left")
            tokenizer.padding_side = "left"
        
        try:
            chat_template_kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
            }
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}], 
                **chat_template_kwargs
            )
            model_thinks = True
        except Exception as e:
            print(f"Model {model_name} does not support thinking.")
            model_thinks = False

        dl = DataLoader(ds, batch_size=32, collate_fn=partial(collate_fn, tokenizer=tokenizer, model_thinks=model_thinks), shuffle=False)


        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device)

        # Evaluation
        correct = 0
        total = 0

        for i, batch_data in tqdm(enumerate(dl)):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=batch_data["input_ids"].to(device),
                    attention_mask=batch_data["attention_mask"].to(device),
                    max_new_tokens=10,
                    do_sample=False,
                )

            output_texts = [tokenizer.decode(outputs[j], skip_special_tokens=False) for j in range(len(batch_data["input_ids"]))]
            model_answers = [extract_first_number(text) for text in output_texts]
            actual_answers = [int(answer) for answer in batch_data["answer"]]

            for j in range(len(batch_data["input_ids"])):
                if model_answers[j] == actual_answers[j]:
                    correct += 1
                total += 1
            
            if i % 50 == 0:
                with open(eval_log_file, "a") as f:
                    f.write(f"correct: {correct}\n")
                    f.write(f"total: {total}\n\n")
                    f.write(f"{output_texts[0]}\n\n")
                    f.write(f"intended answer: {batch_data['answer'][0]}\n")
                    f.write(f"extracted answer: {model_answers[0]}\n")
                    f.write("--------------------------------\n")

        acc = correct / total
        print(f"Accuracy: {acc}")
        
        # Load existing results if they exist
        if os.path.exists(save_dir):
            with open(save_dir, "r") as f:
                results = json.load(f)
        else:
            results = {}
        
        # Update results with current model
        results[model_name] = {
            "accuracy": acc,
            "correct": correct,
            "total": total,
        }

        # Save updated results
        with open(save_dir, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Results saved for {model_name}")
        
        # Clean up model from memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

