from nltk.lm.vocabulary import Vocabulary
from nltk import download
import glob
import itertools
import email
import string
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.corpus import stopwords
from nltk.data import load
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup

download('punkt', download_dir='.')
download('stopwords')

### FUNCIONES
# Función para leer los emails de un directorio dado (num es el número de correos del directorio que se quieren leer)

def readBodies(url):
    url = str(url)
    mails = []
    i = 0
    for s in glob.glob(url+"*"):
        with open(url+s.split("\\")[1]) as file_:
            email_message = email.message_from_file(file_)
            
            for part in email_message.walk():
                body = ''
                clean_body = ''
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True)
                    body= body.decode('latin-1')
                elif part.get_content_type() == "text/html":
                    html_body = part.get_payload(decode=True)
                    body = html_body.decode('latin-1')
                
                htmlParse = BeautifulSoup(body, 'html.parser')
                mails.append(htmlParse.getText())
                
        if i >= 50:
            break
        i=i+1
    return mails
spam_subjects = readBodies("Enron-Spam-Splited/train/no_deseado/") #9494
print(spam_subjects)