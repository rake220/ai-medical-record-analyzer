#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[14]:


import streamlit as st
import PyPDF2
from transformers import pipeline

# Load the medical model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

model = load_model()

st.title("📄 AI-Powered Medical Record Analyzer")
st.write("Upload a medical record (PDF), and the AI will extract insights.")

# File upload
uploaded_file = st.file_uploader("Upload your medical record (PDF)", type=["pdf"])

def extract_text_from_pdf(file):
    # Using PyPDF2 for text extraction
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

if uploaded_file is not None:
    st.subheader("📄 Extracted Text:")
    extracted_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text", extracted_text, height=200)

    if st.button("Analyze Report"):
        with st.spinner("Processing..."):
            prompt = f"Analyze the following medical report and provide insights:\n{extracted_text}"
            try:
                response = model(prompt, max_length=250, do_sample=True)
                st.subheader("🔍 AI Insights:")
                st.write(response[0]['generated_text'])
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.markdown("⚠️ **Disclaimer:** This AI tool is for informational purposes only. Consult a doctor for medical advice.")


# In[ ]:




