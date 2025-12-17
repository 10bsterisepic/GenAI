!pip install transformers datasets rank_bm25 --quiet
abstracts=["Aspirin is used to reduce fever and relieve mild to moderate pain.",
"Ibuprofen is a nonsteroidal anti-inflammatory drug used for treating pain, fever and inflammation.",
"Paracetamol, also known as acetaminophen, is used to treat pain and fever.",
"Metformin is a medication used to treat type 2 diabetes.",
"Atorvastatin is used to prevent cardiovascular disease and treat abnormal lipid levels."]
titles=["Aspirin", "Ibuprofen", "Paracetamol", "Metformin", "Atorvastatin"]

#3. Tokenize abstracts for BM25
import re
def simple_tokenizer(text):
    return re.findall(r"\b\w+\b", text.lower())
from rank_bm25 import BM25Okapi
tokenized_abstracts=[simple_tokenizer(doc) for doc in abstracts]
BM25=BM25Okapi(tokenized_abstracts)

#4. Define biomedical and retrieve top N docs
query="What is used to treat brain tumor?"
tokenized_query=simple_tokenizer(query)
top_n=3
top_docs= BM25.get_top_n(tokenized_query, abstracts, n=top_n)

#5. Load HuggingFace QA Model (DistilBERT or BioBERT)
from transformers import pipeline
qa_pipeline=pipeline('question-answering', model='distilbert-base-cased-distilled-squad')

#6. Run QA over each retrieved document
answers=[]
for i, context in enumerate(top_docs):
    result=qa_pipeline(question=query, context=context)
    answers.append((result['answer'], result['score'], context))

#7. Sort and display best answer
answers.sort(key=lambda x: x[1], reverse=True)
print(f"\nQuery: {query}")
print(f"Best Answer: {answers[0][0]} (Confidence: {answers[0][1]:.2f})")
print(f"Context:\n{answers[0][2]}")
