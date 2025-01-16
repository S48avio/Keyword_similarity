from flask import Flask, request, jsonify
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv("store.env")

# Ensure necessary NLTK packages are downloaded
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
    experience_pattern = re.compile(r'"text":\s?"(.*?)",\s?"label":\s?"EXPERIENCE"')
    skill_pattern = re.compile(r'"text":\s?"(.*?)",\s?"label":\s?"SKILL"')

    experience_entities = experience_pattern.findall(ner_result)
    skill_entities = skill_pattern.findall(ner_result)

    return experience_entities, skill_entities

# Endpoint to process job description and resume
@app.route('https://keyword-similarity.onrender.com/process', methods=['POST'])
def process_data():
    try:
        # Parse JSON payload
        data = request.get_json()
        if not data or 'resume_text' not in data or 'job_description' not in data:
            return jsonify({"error": "Missing required fields: 'resume_text' or 'job_description'"}), 400

        # Extract data from JSON
        resume_text = data.get('resume_text')
        job_description = data.get('job_description')
        shortform_file_path = data.get('shortform_file_path', "shortform.txt")  # Default file path

        # Load shortform dictionary
        shortform_dict = {}
        if os.path.exists(shortform_file_path):
            with open(shortform_file_path, 'r') as file:
                for line in file:
                    if ':' in line:
                        shortform, fullform = line.strip().split(':', 1)
                        shortform_dict[shortform.strip()] = fullform.strip()

        # Preprocess job description and resume text
        cleaned_job_description = preprocess_text(job_description)
        expanded_job_description = replace_shortforms(cleaned_job_description, shortform_dict)

        cleaned_resume_text = preprocess_text(resume_text)
        expanded_resume_text = replace_shortforms(cleaned_resume_text, shortform_dict)

        # Named Entity Recognition for Job Description
        system_prompt_job = f"""
        Please perform Named Entity Recognition (NER) on the following job description, and extract the entities related to EXPERIENCE and SKILL:

        Job Description:{expanded_job_description}

        Return the entities in the following format:
        [
            {{"text": "Experience Text", "label": "EXPERIENCE"}},
            {{"text": "Skill Name", "label": "SKILL"}}
        ]
        """
        response = model.generate_content(system_prompt_job)
        jd_ner_result = response.text
        jd_experience_entities, jd_skill_entities = extract_experience_and_skills(jd_ner_result)

        # Named Entity Recognition for Resume
        system_prompt_resume = f"""
        Perform Named Entity Recognition (NER) on the following resume data:
        {expanded_resume_text}
        Return entities labeled as EXPERIENCE and SKILL.
        """
        response_resume = model.generate_content(system_prompt_resume)
        resume_ner_result = response_resume.text
        resume_experience_entities, resume_skill_entities = extract_experience_and_skills(resume_ner_result)

        # Compute similarity
        skill_similarity = compute_similarity(jd_skill_entities, resume_skill_entities)

        # Compute scores
        experience_count = len(resume_experience_entities)
        project_count = len(resume_skill_entities)
        experience_score = 0.5 * experience_count
        project_score = 0.3 * project_count
        total_score = experience_score + project_score + skill_similarity

        return jsonify({
            "experience_score": experience_score,
            "project_score": project_score,
            "skill_similarity": skill_similarity,
            "total_score": total_score
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Function to compute similarity between job description and resume skills
def compute_similarity(jd_skill_entities, resume_skill_entities):
    jd_skill_text = " ".join(jd_skill_entities)
    resume_skill_text = " ".join(resume_skill_entities)
    if not jd_skill_text or not resume_skill_text:
        return 0
    vectorizer = CountVectorizer(stop_words='english')
    jd_vector = vectorizer.fit_transform([jd_skill_text])
    resume_vector = vectorizer.transform([resume_skill_text])
    return cosine_similarity(jd_vector, resume_vector)[0, 0]

if __name__ == '__main__':
    app.run(debug=True)
