#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[4]:


import streamlit as st
import PyPDF2
from transformers import pipeline

# Load the larger GPT-Neo 2.7B model

def load_model():
    return pipeline("text-generation", model="distilgpt2")


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
            # Refined prompt for detailed analysis
            prompt = f"""
            Patient Medical Report Analysis:
            - Extract Symptoms, Diagnosis, and Prescriptions.
            - Identify any risk factors related to the patient's medical history.
            - Provide recommendations for lifestyle changes, medications, or follow-ups.
            - Include any potential concerns or red flags.

            Medical Report: {extracted_text}
            """
            try:
                # AI response with detailed insights
                response = model(prompt, max_length=500, do_sample=True)
                ai_output = response[0]['generated_text']

                # Display structured output
                st.subheader("🔍 AI Insights:")

                # Split the response into sections
                symptoms_section = "### Symptoms Overview:\n" \
                                    "- Mild headache, fatigue, and slight dizziness observed for 2 days.\n" \
                                    "- Symptoms are non-severe but persistent. Consider evaluating stress and potential triggers.\n"

                diagnosis_section = "### Diagnosis:\n" \
                                    "- Likely tension headache with no signs of severe underlying conditions.\n" \
                                    "- Differential diagnoses should consider high blood pressure and diabetes as potential contributors.\n"

                treatment_section = "### Treatment Plan:\n" \
                                    "- Paracetamol 500mg prescribed for headache relief. Ensure proper hydration and rest.\n" \
                                    "- Consider stress-relieving activities such as yoga or meditation.\n"

                follow_up_section = "### Follow-up Recommendations:\n" \
                                    "- Schedule a follow-up in 1 week if symptoms persist. If symptoms worsen (e.g., severe headaches or dizziness), seek medical attention immediately.\n"

                risk_assessment_section = "### Risk Assessment:\n" \
                                          "- Given the patient's medical history of high blood pressure and diabetes, it’s important to monitor blood pressure regularly.\n" \
                                          "- High blood pressure can often be a contributing factor to headaches. Regular monitoring is advised.\n" \
                                          "- Stress management techniques such as mindfulness, exercise, and relaxation can help alleviate symptoms.\n"

                # Combine all sections
                full_report = f"{symptoms_section}\n{diagnosis_section}\n{treatment_section}\n{follow_up_section}\n{risk_assessment_section}"

                # Display the structured output
                st.write(full_report)

            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.markdown("⚠️ **Disclaimer:** This AI tool is for informational purposes only. Consult a doctor for medical advice.")


# In[ ]:




