# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:50:11 2024

@author: shiv
"""


# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import click
import spacy
import docx2txt
import pdfplumber
from pickle import load
import requests
import re
import os
import sklearn
import PyPDF2
import nltk
import pickle as pk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import matplotlib.pyplot  as plt
stop=set(stopwords.words('english'))
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)
from sklearn.feature_extraction.text import TfidfVectorizer

#----------------------------------------------------------------------------------------------------

st.title('             RESUME CLASSIFICATION WEBAPP    ')
st.markdown('<style>h1{color: Purple;}</style>', unsafe_allow_html=True)
st.markdown('This is a very simple webapp for classifying resumes :chart:')
st.subheader("Hello 👋")

# FUNCTIONS
def extract_skills(resume_text):

    nlp_text = nlp(resume_text)
    noun_chunks = nlp_text.noun_chunks

    tokens = [token.text for token in nlp_text if not token.is_stop] # removing stop words and implementing word tokenization
            
    
    #data = pd.read_csv('C:\\Users\\shiv\\Desktop\\P376\\dummy\\skills.csv') # reading the csv file
            
    
    #skills = list(data.columns.values)# extract values
    skills=[
        'react js', 'python' , 'javascript', 'html', 'css', 'webpack', 'npm',
         'hcm', 'report writer', 'eib', 'core connector',
        'sql', 'database', 'pl/sql', 'oracle', 'mysql', 't-sql'
        , 'hrms', 'peoplecode', 'application engine', 'sqr',
      ]                
    skillset = []
            
    
    for token in tokens:                 # check for one-grams (example: python)
        if token.lower() in skills:
            skillset.append(token)
            
   
    for token in noun_chunks:            # check for bi-grams and tri-grams (example: machine learning)
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
            
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

## Function to extract experience from text
def extract_experience(text):
    doc = nlp(text)
    experience = ''
    
    ## Extract experience using spaCy's entity recognition and dependency parsing
    for ent in doc.ents:
        if ent.label_ == "DATE" and 'year' in ent.text.lower():
            ## Extract the number of years from the entity text
            for token in doc:
                if token.idx >= ent.start_char and token.idx + len(token) <= ent.end_char:
                    if token.pos_ == 'NUM':
                        experience = token.text
                        break
    
    return experience



def getText(filename):
      
    # Create empty string 
    fullText = ''
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        
        for para in doc:
            fullText = fullText + para
            
           
    else:  
        with pdfplumber.open(filename) as pdf_file:
            pdoc = PyPDF2.PdfFileReader(filename)
            number_of_pages = pdoc.getNumPages()
            page = pdoc.pages[0]
            page_content = page.extractText()
             
        for paragraph in page_content:
            fullText =  fullText + paragraph
         
    return (fullText)


def display(doc_file):
    resume = []
    if doc_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))

    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
            
    return resume


def preprocess(sentence):
    sentence=str(sentence)
    sentence = sentence.lower()
    sentence=sentence.replace('{html}',"") 
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url=re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)  
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return " ".join(lemma_words) 

file_type=pd.DataFrame([], columns=['Uploaded File',  'Predicted Profile','experience'])
filename = []
predicted = []
Skills = []
exp=[]
#-------------------------------------------------------------------------------------------------
# MAIN CODE
import pickle as pk
model = pk.load(open('C:\\Users\\shiv\\Desktop\\P376\\dummy\\Model_DT.pkl', 'rb'))
Vectorizer = pk.load(open('C:\\Users\\shiv\\Desktop\\P376\\dummy\\tfidf_vector.pkl', 'rb'))

upload_file = st.file_uploader('Please Upload Your Resumes',
                                type= ['docx','pdf'],accept_multiple_files=True)
  
for doc_file in upload_file:
    if doc_file is not None:
        filename.append(doc_file.name)
        cleaned=preprocess(display(doc_file))
        prediction = model.predict(Vectorizer.transform([cleaned]))[0]
        predicted.append(prediction)
        extText = getText(doc_file)
        Skills.append(extract_skills(extText))
        exp.append(extract_experience(extText))
        
if st.button('category'):
    file_type['Uploaded File'] = filename
    #file_type['Skills'] = skills
    file_type['Predicted Profile'] = predicted
    file_type['experience'] = exp
    st.table(file_type.style.format())
    
if st.button('skills'):
    skill_type=pd.DataFrame([], columns=['Uploaded File','Skills',])
    #filename = []
    #skills = []
    skill_type['Uploaded File'] = filename
    skill_type['Skills'] = Skills
    st.table(skill_type.style.format())
    