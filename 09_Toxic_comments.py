# Install compatible versions
!pip install -q numpy==1.26.4 # Ensure NumPy 1.x for compatibility with torch
!pip install -q torch==2.2.2 torchvision torchaudio
!pip install -q transformers==4.40.2
import torch
from transformers import BertTokenizer, BertForSequenceClassification
texts = [
    "I hate you",
    "You are stupid",
    "Let's be friends",
    "I love this",
    "Go away idiot",
    "You are amazing",
    "Such a dumb idea",
    "Great work team"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0]
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=32)
input_ids = encodings["input_ids"]
attention_mask = encodings["attention_mask"]
labels = torch.tensor(labels)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()
model.train()
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(input_ids, attention_mask=attention_mask)
    loss = loss_fn(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")
model.eval()
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    preds = torch.argmax(outputs.logits, dim=1)

for text, label, pred in zip(texts, labels, preds):
    print(f"Text: {text}\nTrue Label: {label}, Predicted: {pred}\n")

# Uninstall and reinstall the transformers library to ensure a clean and compatible version.
# Removing the version constraint to allow pip to install the latest stable version.
!pip uninstall -y transformers
!pip install transformers

from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="Hate-speech-CNERG/bert-base-uncased-hatexplain"
)
comments = [
    "You are an idiot.",
    "I hope you have a wonderful day!",
    "People like you should not exist.",
    "This is a normal comment."
]

for c in comments:
    result = classifier(c)[0]
    print(f"Text: {c}")
    if result['score']<0.75:
        print(f"Score: {result['score']}\ntoxic")
    else:
        print(f"Score: {result['score']}")
    print("-" * 40)
