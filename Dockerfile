# Use a slim Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application files to the container
COPY final.py store.env shortform.txt multi_word_phrases.txt ./

# Install Python dependencies
RUN pip install flask nltk scikit-learn PyPDF2 google-generativeai python-dotenv

# Download necessary NLTK data
RUN python -m nltk.downloader punkt wordnet

# Expose the port the app runs on
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=final.py

# Command to run the application
CMD ["flask", "run", "--host=0.0.0.0"]
