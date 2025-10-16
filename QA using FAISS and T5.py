pip install faiss-cpu transformers sentence-transformers
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
t5_tok=T5Tokenizer.from_pretrained('google/flan-t5-base')
t5_model=T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

embed_model=SentenceTransformer('all-MiniLM-L6-v2')

legal_docs=[ "The Supreme Court held that the right to privacy is a fundamental right under Article 21 of the Indian Constitution.",
            "In Kesavananda Bharati case, the Supreme Court ruled that the basic structure of the Constitution cannot be altered by amendments.",
            "The court emphasized the importance of due process in arrest under the Code of Criminal Procedure, 1973.",
             "The Right to Freedom of speech is subject to reasonable restrictions under Article 19(2)."
             ]

#embed docs
doc_embeddings=embed_model.encode(legal_docs)

#create FAISS index
dimension=doc_embeddings.shape[1]
faiss_index=faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(doc_embeddings))

#user query
query="What is the importance of the right to privacy in Indian law?"
print("Query:",query,"\n")

#encode query
query_embedding = embed_model.encode([query])
D, I = faiss_index.search(np.array(query_embedding), k=2)

#fetch relevant documents
retrieved_docs = [legal_docs[i] for i in I[0]]

print("Retrieved Legal Documents:")
for doc in retrieved_docs:
    print("-", doc)

#combine and use T5 for answer generation
input_text = " ".join(retrieved_docs)
t5_input = f"question: {query} context: {input_text}"
inputs = t5_tok(t5_input, return_tensors="pt", max_length=512, truncation=True)

#generate answer
outputs = t5_model.generate(**inputs, max_new_tokens=100)
answer = t5_tok.decode(outputs[0], skip_special_tokens=True)

print("\nGenerated Answer:")
print(answer)

#multi-level summarization
summary_input = f"summarize: {input_text}"
inputs = t5_tok(summary_input, return_tensors="pt", max_length=512, truncation=True)

summary_outputs = t5_model.generate(**inputs, max_new_tokens=100)
summary = t5_tok.decode(summary_outputs[0], skip_special_tokens=True)

print("\nGenerated Summary:")
print(summary)
