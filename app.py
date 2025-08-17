# Required installations:
# pip install streamlit
# pip install scikit-learn
# pip install python-docx
# pip install PyPDF2

import streamlit as st
import pickle
import docx   # Extract text from Word file
import PyPDF2 # Extract text from PDF
import re

# Load pre-trained model, TF-IDF vectorizer, and encoder
svc_clf = pickle.load(open('clf.pkl', 'rb'))
tfidf_vec = pickle.load(open('tfidf.pkl', 'rb'))
label_enc = pickle.load(open('encoder.pkl', 'rb'))

# Function to clean resume text
def clean_text(raw_txt):
    txt = re.sub(r'http\S+\s', ' ', raw_txt)
    txt = re.sub(r'RT|cc', ' ', txt)
    txt = re.sub(r'#\S+\s', ' ', txt)
    txt = re.sub(r'@\S+', ' ', txt)
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub(r'\s+', ' ', txt)
    return txt

# Extract text from PDF
def extract_pdf(file):
    reader = PyPDF2.PdfReader(file)
    txt = ''
    for page in reader.pages:
        txt += page.extract_text()
    return txt

# Extract text from DOCX
def extract_docx(file):
    document = docx.Document(file)
    txt = ''
    for para in document.paragraphs:
        txt += para.text + '\n'
    return txt

# Extract text from TXT with encoding handling
def extract_txt(file):
    try:
        txt = file.read().decode('utf-8')
    except UnicodeDecodeError:
        txt = file.read().decode('latin-1')
    return txt

# Handle file upload and extract text
def extract_resume(uploaded_file):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext == 'pdf':
        return extract_pdf(uploaded_file)
    elif file_ext == 'docx':
        return extract_docx(uploaded_file)
    elif file_ext == 'txt':
        return extract_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type (Only: PDF, DOCX, TXT).")

# Predict Category for a resume
def predict_category(input_resume):
    # Clean text
    cleaned_txt = clean_text(input_resume)
    # TF-IDF features
    vec_txt = tfidf_vec.transform([cleaned_txt])
    vec_txt = vec_txt.toarray()
    # Predict
    pred_cat = svc_clf.predict(vec_txt)
    pred_cat_name = label_enc.inverse_transform(pred_cat)
    return pred_cat_name[0]

# Streamlit app layout
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")

    st.title("Resume Category Prediction App")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_txt = extract_resume(uploaded_file)
            st.success("Successfully extracted text from resume ✅")

            # Optionally show the text
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_txt, height=300)

            # Prediction
            st.subheader("Predicted Category")
            category = predict_category(resume_txt)
            st.write(f"The predicted category is: **{category}**")

        except Exception as e:
            st.error(f"Error processing file ❌: {str(e)}")

if __name__ == "__main__":
    main()
