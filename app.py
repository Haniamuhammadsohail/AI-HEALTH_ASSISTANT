import streamlit as st
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration

# Initialize the model
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)

# Streamlit UI
st.title("AI Health Assistant (RAG-based)")

def get_answer_rag(question):
    inputs = tokenizer(question, return_tensors="pt")
    retrieved_docs = retriever.retrieve(inputs['input_ids'], top_k=3)
    outputs = model.generate(input_ids=inputs['input_ids'], context_input_ids=retrieved_docs['input_ids'])
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# Ask the user for a health-related question
question = st.text_input("Ask a health-related question:")
if question:
    answer = get_answer_rag(question)
    st.write(f"Answer: {answer}")
