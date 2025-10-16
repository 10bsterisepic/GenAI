!pip install transformers datasets --quiet
from transformers import pipeline
from datasets import load_dataset
import numpy as np

#load sentiment-analysis pipeline with DistilBERT
sentiment_analyzer=pipeline('sentiment-analysis') #(model='nlptown/bert-base-multilingual-uncased-sentiment')
dataset=load_dataset('imdb', split='test[:5]')

results=[]

for review in dataset:
    text=review['text']
    prediction=sentiment_analyzer(text[:512])[0] #truncate to 512 tokens
    results.append({
        'text':text[:100]+'...',
        'actual':'POSITIVE' if review['label']==1 else 'NEGATIVE',
        'predicted':prediction['label'],
        'confidence':round(prediction['score'], 4)
    })

for i, r in enumerate(results, 1):
    print(f"\nReview {i}:")
    print(f"Actual:' {r['actual']}")
    print(f"Predicted:' {r['predicted']}")
    print(f"Confidence:' {r['confidence']}")
    print(f"Review Snippet: {r['text']}")
