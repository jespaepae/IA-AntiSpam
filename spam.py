from nltk.lm.vocabulary import Vocabulary
from nltk import download
import itertools
from lectura import readEmails, cleanText, read_email
from filtro_naive_bayes import entrenar_modelo, clasificador_nb
from filtro_tf_idf import entrenar_tf_idf
import numpy as np
download('punkt', download_dir='.')
download('stopwords')

spam_subjects = readEmails("Enron-Spam-Splited/train/no_deseado/") #9494
ham_subjects = readEmails("Enron-Spam-Splited/train/legítimo/")   #10828
clean_spam_subjects = [cleanText(s) for s in spam_subjects]
clean_ham_subjects = [cleanText(s) for s in ham_subjects]
clean_subjects = np.concatenate((clean_spam_subjects, clean_ham_subjects), axis=0)

entrenar_modelo(clean_subjects)
result = clasificador_nb("Enron-Spam-Splited/train/no_deseado/0")
print(result)
#entrenar_tf_idf(clean_subjects)
