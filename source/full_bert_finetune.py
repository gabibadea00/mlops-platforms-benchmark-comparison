import os
import shutil
import torch
import evaluate
from transformers import (
    BertTokenizerFast,
    BertForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import load_dataset
from transformers.trainer_utils import EvalPrediction

# Verbose print
def log(msg):
    print(f"[INFO] {msg}")

# ==== 1. Load SQuAD v2 Dataset ====
log("Loading SQuAD v2.0 dataset...")
dataset = load_dataset("squad_v2")

# ==== 2. Load Tokenizer and Model ====
model_name = "bert-base-uncased"
log(f"Downloading and loading tokenizer & model: {model_name}")
tokenizer = BertTokenizerFast.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# ==== 3. Preprocess Function ====
log("Preprocessing dataset...")

def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    inputs = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length"
    )

    sample_mapping = inputs.pop("overflow_to_sample_mapping")
    offset_mapping = inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = inputs["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        sequence_ids = inputs.sequence_ids(i)
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]

        if len(answers["answer_start"]) == 0:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        answer_start = answers["answer_start"][0]
        answer_text = answers["text"][0]

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if not (offsets[token_start_index][0] <= answer_start and offsets[token_end_index][1] >= answer_start + len(answer_text)):
            start_positions.append(cls_index)
            end_positions.append(cls_index)
        else:
            start = token_start_index
            while start < len(offsets) and offsets[start][0] <= answer_start:
                start += 1
            end = token_end_index
            while end >= 0 and offsets[end][1] >= answer_start + len(answer_text):
                end -= 1
            start_positions.append(start - 1)
            end_positions.append(end + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

encoded_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# ==== 4. Define Training Arguments ====
log("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./bert-squad2-output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    report_to="none"
)

# ==== 5. Define Metric ====
log("Loading SQuAD evaluation metric...")
metric = evaluate.load("squad_v2")

def compute_metrics(p: EvalPrediction):
    predictions = p.predictions
    start_logits, end_logits = predictions
    predictions = []

    for i in range(len(start_logits)):
        start_idx = start_logits[i].argmax()
        end_idx = end_logits[i].argmax()
        predictions.append({
            "id": p.label_ids["id"][i],
            "prediction_text": tokenizer.decode(p.label_ids["input_ids"][i][start_idx:end_idx + 1]),
            "no_answer_probability": 0.0
        })

    references = [{"id": id_, "answers": ans} for id_, ans in zip(p.label_ids["id"], p.label_ids["answers"])]
    return metric.compute(predictions=predictions, references=references)

# ==== 6. Train the Model ====
log("Starting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=None  # set to compute_metrics for full evaluation
)

trainer.train()

# ==== 7. Evaluate the Model ====
log("Evaluating model...")
results = trainer.evaluate()
for key, value in results.items():
    print(f"{key}: {value:.4f}")

# ==== 8. Cleanup Hugging Face cache ====
log("Cleaning up Hugging Face cache...")

hf_cache_dir = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(hf_cache_dir):
    shutil.rmtree(hf_cache_dir)
    log("Deleted Hugging Face cache.")

log("All done!")