# Resume Screening App

![Alt text](resume-screening-software.png)


A Streamlit web application that predicts the job category of a resume based on its content. Users can upload resumes in PDF, DOCX, or TXT formats, and the app extracts, cleans, and processes the resume text to predict the most suitable job category using a pre-trained machine learning model.

---

## Features

- Supports resume upload in PDF, DOCX, and TXT file formats.
- Automatically extracts and cleans resume text.
- Uses TF-IDF vectorization and a Support Vector Classifier (SVC) for job category prediction.
- Highly accurate classification model trained on a balanced dataset of 25 job categories.
- Interactive and user-friendly interface built with Streamlit.
- Option to view extracted resume text for verification.

---

## Dataset

- The dataset contains 962 resumes labeled across 25 job categories such as Data Science, Java Developer, Network Security Engineer, Advocate, Health and Fitness, and others.
- Initial dataset was imbalanced; oversampling was applied to equalize category sample size (each category contains 84 samples).
- Categories and resume texts are stored in `UpdatedResumeDataSet.csv`.

---

## Data Preprocessing

- Resume texts are cleaned to remove URLs, hashtags, mentions, special characters, and non-ASCII symbols.
- Texts are vectorized using TF-IDF with English stop words removal.
- Categories are label encoded into numeric format for model training.

---

## Model Training and Evaluation

- The dataset is split into 80% training and 20% testing.
- Multiple classifiers were evaluated with the One-vs-Rest approach:
  - K-Nearest Neighbors (KNN) — Accuracy: ~99.5%
  - Support Vector Classifier (SVC) — Accuracy: 100%
  - Random Forest Classifier — Accuracy: 100%
- Confusion matrices and classification reports demonstrate excellent model performance.

---

## Installation

Ensure you have Python 3.x installed. Install dependencies using:

`pip install -r requirements.txt`


Required key packages:

- streamlit
- scikit-learn
- python-docx
- PyPDF2
- pandas
- numpy

Full package list is provided in `requirements.txt`.

---

## Usage

1. Clone this repository.
2. Place the following files in the root directory:
   - `clf.pkl` (trained classifier)
   - `tfidf.pkl` (TF-IDF vectorizer)
   - `encoder.pkl` (label encoder)
3. Run the Streamlit app:

`streamlit run app.py`


4. In the web interface, upload a resume file (PDF, DOCX, or TXT).
5. View the predicted job category.
6. Optionally, toggle "Show extracted text" to verify text extraction.

---

## How It Works

- The uploaded resume is parsed and text is extracted based on file type.
- The extracted text is cleaned and preprocessed.
- The cleaned text is transformed into TF-IDF features.
- The trained SVC predicts the job category.
- Predicted category label is decoded to original category name and displayed.

---

## Example Prediction

`my_resume = “”“I am a data scientist specializing in machine learning, deep learning, and computer vision…””” predicted_category = predict_category(my_resume) print(predicted_category)  # Output: Data Science`


---

## File Format Support

- PDF (.pdf)
- Word Document (.docx)
- Plain Text (.txt)

---

## Contact

For any queries or suggestions, contact:

- Email: aditisahu24@iitk.ac.in
- GitHub: https://github.com/aditisahu911

