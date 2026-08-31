import os
import torch
from datasets import load_from_disk
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import evaluate
from config import get_training_config
from dataclasses import dataclass
from typing import Any, Dict, List, Union

MODEL_ID = "openai/whisper-small"
cfg = get_training_config()

feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
tokenizer = WhisperTokenizer.from_pretrained(MODEL_ID, language="marathi", task="transcribe")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="marathi", task="transcribe")

print("Loading dataset...")
dataset = load_from_disk("/workspace/outputs/konkani_dataset")

def prepare(batch):
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = tokenizer(batch["transcript"]).input_ids
    return batch

print("Preprocessing (this may take a few minutes)...")
dataset = dataset.map(prepare, remove_columns=dataset["train"].column_names, num_proc=1)

print("Loading base model...")
quantization_config = BitsAndBytesConfig(load_in_8bit=cfg["use_8bit"])
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, quantization_config=quantization_config, device_map="auto"
)
model = prepare_model_for_kbit_training(model)

# Conservative LoRA settings — smaller rank + higher dropout than usual,
# since we only have ~1.7 hours of actual training audio (high overfitting risk)
lora_config = LoraConfig(
    r=16,                    # lower rank than my original plan (was 32) — less capacity to overfit
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,        # higher than usual — small data needs more regularization
    bias="none",
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False
model.print_trainable_parameters()

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def compute_metrics(pred):
    pred_ids, label_ids = pred.predictions, pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    return {
        "wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": 100 * cer_metric.compute(predictions=pred_str, references=label_str),
    }

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

training_args = Seq2SeqTrainingArguments(
    output_dir="/workspace/outputs/whisper-small-konkani-lora",
    per_device_train_batch_size=cfg["batch_size"],
    gradient_accumulation_steps=cfg["grad_accum"],
    learning_rate=1e-3,
    warmup_steps=30,
    num_train_epochs=6,          # fewer epochs than original plan — small data overfits fast
    fp16=cfg["fp16"],
    per_device_eval_batch_size=cfg["batch_size"],
    eval_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    predict_with_generate=True,
    generation_max_length=128,
    report_to=["none"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

print("Starting training...")
trainer.train()

print("Saving final adapter...")
model.save_pretrained("/workspace/outputs/whisper-small-konkani-lora-adapter")
processor.save_pretrained("/workspace/outputs/whisper-small-konkani-lora-adapter")
print("Done.")
