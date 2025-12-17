!pip -q install transformers datasets peft accelerate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

ds = load_dataset("ag_news")
num_labels = 4

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

ds = ds.map(tokenize, batched=True)
base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query","value"],  # attention projections
    task_type="SEQ_CLS"
)

model = get_peft_model(base_model, config)

args = TrainingArguments(
    output_dir="out",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=50,
    save_strategy="no",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"].shuffle(seed=42).select(range(2000)),  # subset for speed
    eval_dataset=ds["test"].select(range(1000)),
    tokenizer=tokenizer
)
trainer.train()
print(trainer.evaluate())
pred = trainer.predict(ds["test"].select(range(5)))
print("Predicted labels:", pred.predictions.argmax(-1))
