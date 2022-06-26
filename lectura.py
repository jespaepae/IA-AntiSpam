from nltk.lm.vocabulary import Vocabulary
from nltk import download
import glob
import itertools
import email
import string
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.data import load
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
download('punkt', download_dir='.')
download('stopwords')

### FUNCIONES
# Función para leer los emails de un directorio dado (num es el número de correos del directorio que se quieren leer)

def readEmails(url):
    url = str(url)
    mails = []
    i = 0
    for s in glob.glob(url+"*"):
        with open(url+s.split("\\")[1]) as file_:
            mail = email.message_from_file(file_)
            mails.append(mail.get("Subject"))
        
    return mails

# Función para leer un email dado una url
def read_email(url):
    subject = []
    with open(url) as file_:
            mail = email.message_from_file(file_)
            subject.append(mail.get("Subject"))
    return subject
def readEmailsWithBody(url):
    url = str(url)
    mails = []
    i = 0
    for s in glob.glob(url+"*"):
        with open(url+s.split("\\")[1], encoding='unicode_escape', errors='ignore') as file_:
            mail = email.message_from_file(file_)
            
            text = mail.get("Subject")
            if text is None:
                text=''
            for part in mail.walk():
                body = ''
                clean_body = ''
                
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True)
                    body= body.decode('latin-1')
                    htmlParse = BeautifulSoup(body, 'html.parser')
                    if htmlParse.getText() is not None:
                        text=text+htmlParse.getText()
                elif part.get_content_type() == "text/html":
                    html_body = part.get_payload(decode=True)
                    body = html_body.decode('latin-1')
                    htmlParse = BeautifulSoup(body, 'html.parser')
                    if htmlParse.getText() is not None:
                        text=text+htmlParse.getText()
        if i>=100:
            break
        i+=1
        
        
                
        mails.append(text)
    return mails

# Función para leer un email dado una url
def read_email(url):
    subject = []
    with open(url) as file_:
            mail = email.message_from_file(file_)
            subject.append(mail.get("Subject"))
    return subject
# Función para eliminar los signos de puntuación de un texto dado

def cleanText(text):
    punctuations = []
    words = ''
    if text != None:
        words = []
        for c in text:
            if c not in string.punctuation:

                punctuations.append(c)
        punctuations = ''.join(punctuations)

        for t in punctuations.split():
            if t.lower() not in stopwords.words('english'):

                words.append(t.lower())
        words = ' '.join(words)
    return words