import os
import shutil
import torch
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator
)
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

# Verbose logging
def log(msg):
    print(f"[INFO] {msg}")

# ==== 1. Load SQuAD v2.0 dataset ====
log("Loading SQuAD v2.0 dataset...")
dataset = load_dataset("squad_v2")

# ==== 2. Load quantized model & tokenizer ====
model_id = "bert-base-uncased"
log(f"Loading tokenizer and quantized model: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForQuestionAnswering.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto"
)

# ==== 3. Apply LoRA configuration ====
log("Applying LoRA configuration...")
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.QUESTION_ANSWERING,
)

model = get_peft_model(model, peft_config)

# ==== 4. Preprocess Function ====
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

# ==== 5. Training Arguments ====
log("Configuring training arguments...")
training_args = TrainingArguments(
    output_dir="./lora-bert-squad2-output",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=50,
    report_to="none"
)

# ==== 6. Define Trainer ====
log("Starting PEFT training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

trainer.train()

# ==== 7. Evaluate ====
log("Evaluating the PEFT model...")
results = trainer.evaluate()
for key, value in results.items():
    print(f"{key}: {value:.4f}")

# ==== 8. Cleanup Hugging Face cache ====
log("Cleaning up Hugging Face cache...")
hf_cache = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(hf_cache):
    shutil.rmtree(hf_cache)
    log("Deleted Hugging Face cache.")

log("✅ LoRA fine-tuning on SQuAD v2.0 complete!")