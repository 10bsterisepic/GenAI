!pip uninstall -y transformers sentences-trasformers huggingface_hb
!pip install transformers==4.41.2 sentence-transformers==3.0.1
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

texts = [
    "Immigrants are stealing jobs.",
    "Women are great leaders.",
    "The policy will improve equality.",
]

labels = ["biased", "neutral"]

for t in texts:
    result = classifier(t, candidate_labels=labels)
    print(f"Text: {t}")
    print(f"Predicted: {result['labels'][0]} | Score: {result['scores'][0]:.4f}")
    print("-" * 40)
