import os
import re

import nltk
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv("store.env")


# Ensure you download the necessary NLTK packages
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Set up the API key and model configuration
genai.configure(api_key=os.getenv("GENAI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize NLTK objects
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to preprocess text (remove stopwords, lemmatize)
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_words]
    return ' '.join(lemmatized_tokens)

# Function to replace shortforms with fullforms in text
def replace_shortforms(text, shortform_dict):
    for shortform, fullform in shortform_dict.items():
        text = re.sub(r'\b' + re.escape(shortform) + r'\b', fullform, text)
    return text

# Function to extract EXPERIENCE and SKILL from the NER result
def extract_experience_and_skills(ner_result):
    experience_entities = []
    skill_entities = []

    # Regex pattern to match the entity labels and their corresponding text
    experience_pattern = re.compile(r'"text":\s?"(.*?)",\s?"label":\s?"EXPERIENCE"')
    skill_pattern = re.compile(r'"text":\s?"(.*?)",\s?"label":\s?"SKILL"')

    # Extract entities using regex
    experience_entities = experience_pattern.findall(ner_result)
    skill_entities = skill_pattern.findall(ner_result)

    return experience_entities, skill_entities

# Function to load shortforms from a text file
def load_shortforms(file_path):
    shortform_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            if ':' in line:
                shortform, fullform = line.strip().split(':', 1)
                shortform_dict[shortform.strip()] = fullform.strip()
    return shortform_dict

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Endpoint to process job description and resume
@app.route('/process', methods=['POST'])
def process_data():
    try:
        # Validate if both 'resume' and 'job_description' are provided
        if 'resume' not in request.files or 'job_description' not in request.form:
            return jsonify({"error": "Missing resume or job description"}), 400

        # Get the uploaded resume file and job description
        resume_file = request.files['resume']
        job_description = request.form['job_description']
       
        # Check if the uploaded file is a PDF
        if not resume_file.filename.endswith('.pdf'):
            return jsonify({"error": "The uploaded resume must be a PDF file"}), 400

        # Extract text from the uploaded resume
        try:
            resume_text = extract_text_from_pdf(resume_file)
        except Exception as e:
            return jsonify({"error": f"Failed to extract text from resume: {str(e)}"}), 500

        # Load shortforms from file
        shortform_dict = load_shortforms("shortform.txt")

        # Preprocess the job description
        cleaned_job_description = preprocess_text(job_description)
        expanded_job_description = replace_shortforms(cleaned_job_description, shortform_dict)

        # Modified System Prompt for NER on Job Description
        system_prompt_job = f"""
        Please perform Named Entity Recognition (NER) on the following job description, and extract the entities related to EXPERIENCE and SKILL:

        Job Description:{expanded_job_description}

        Return the entities in the following format:
        [
            {{"text": "Experience Text", "label": "EXPERIENCE"}},
            {{"text": "Skill Name", "label": "SKILL"}}
        ]
        Ensure that you capture the key job experiences and skills from the input text.
        """

        response = model.generate_content(system_prompt_job)
        jd_ner_result = response.text

        # Extract entities for experience and skills from job description
        jd_experience_entities, jd_skill_entities = extract_experience_and_skills(jd_ner_result)

        # Preprocess and expand the resume text
        cleaned_resume_text = preprocess_text(resume_text)
        expanded_resume_text = replace_shortforms(cleaned_resume_text, shortform_dict)

        # System prompt for NER on resume
        system_prompt_resume = f"""
        Please perform Named Entity Recognition (NER) on the following resume data:

        {expanded_resume_text}

        Return the entities with the following labels: PERSON, ORGANIZATION (ORG), LOCATION (GPE), PROJECT, EXPERIENCE, SKILL, and any other relevant information. Format the output as a list of dictionaries, where each dictionary contains "text" and "label" keys, like this:
        [{{"text": "Savio Sunny", "label": "PERSON"}},
        {{"text": "RSET", "label": "ORG"}},
        {{"text": "Kochi", "label": "GPE"}},
        {{"text": "Python", "label": "SKILL"}},
        {{"text": "Personalized Hospital Appointment Management System", "label": "PROJECT"}},
        {{"text": "Google Developer Student Club", "label": "ORG"}},
        {{"text": "Backend Developer Intern", "label": "EXPERIENCE"}}]

        Also, please return the counts of the entities labeled "EXPERIENCE" and "PROJECT" in the following format:

        [
            {{"text": "experience_count", "label": "count"}},
            {{"text": "project_count", "label": "count"}}
        ]
        """

        response_resume = model.generate_content(system_prompt_resume)
        resume_ner_result = response_resume.text

        # Extract experience and skills from the resume NER response
        resume_experience_entities, resume_skill_entities = extract_experience_and_skills(resume_ner_result)

        # Extract counts from the NER response text
        experience_count = resume_ner_result.count("EXPERIENCE")
        project_count = resume_ner_result.count("PROJECT")
        print(experience_count)
        print(project_count)

        # Compute the weighted score
        experience_score = 0.5 * experience_count
        project_score = 0.3 * project_count
        

        # Compute similarity scores for skills
        skill_similarity = compute_similarity(jd_skill_entities, resume_skill_entities)
        total_score = experience_score + project_score+skill_similarity

        # Return the results as JSON
        return jsonify({
            "experience_score": experience_score,
            "project_score": project_score,
            "total_score": total_score,
            "skill_similarity": skill_similarity
            
        })

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


# Function to compute similarity between job description and resume skills
def compute_similarity(jd_skill_entities, resume_skill_entities):
    # Concatenate entities into strings
    jd_skill_text = " ".join(jd_skill_entities)
    resume_skill_text = " ".join(resume_skill_entities)
   
    # Calculate the cosine similarity
    if not jd_skill_text or not resume_skill_text:
        skill_similarity = 0
    else:
        vectorizer_skill = CountVectorizer(stop_words='english')
        jd_skill_vector = vectorizer_skill.fit_transform([jd_skill_text])
        resume_skill_vector = vectorizer_skill.transform([resume_skill_text])
        skill_similarity = cosine_similarity(jd_skill_vector, resume_skill_vector)[0, 0]

    return skill_similarity

if __name__ == '__main__':
    app.run(debug=True)
