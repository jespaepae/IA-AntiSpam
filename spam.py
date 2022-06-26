from nltk.lm.vocabulary import Vocabulary
from nltk import download
import itertools
from lectura import readEmails, cleanText, read_email, readEmailsWithBody
from filtro_naive_bayes import entrenar_modelo, clasificador_nb
from filtro_tf_idf import entrenar_tf_idf, clasificador_tf_idf
from filtro_tf_idf import entrenar_tf_idf
import numpy as np
import gc
download('punkt', download_dir='.')
download('stopwords')

spam_subjects = readEmails("Enron-Spam-Splited/train/no_deseado/") #9494
ham_subjects = readEmails("Enron-Spam-Splited/train/legítimo/")   #10828
#spam_subjects = readEmailsWithBody("Enron-Spam-Splited/train/no_deseado/") #9494
#ham_subjects = readEmailsWithBody("Enron-Spam-Splited/train/legítimo/")   #10828
clean_spam_subjects = [cleanText(s) for s in spam_subjects]
clean_ham_subjects = [cleanText(s) for s in ham_subjects]
clean_subjects = np.concatenate((clean_spam_subjects, clean_ham_subjects), axis=0)
print(clean_subjects)
entrenar_modelo(clean_subjects, len(clean_spam_subjects), len(clean_ham_subjects))
result = clasificador_nb("Enron-Spam-Splited/train/no_deseado/0")
print(result)
entrenar_tf_idf(clean_subjects, len(clean_spam_subjects), len(clean_ham_subjects))

result = clasificador_tf_idf("Enron-Spam-Splited/train/no_deseado/0")
print(result)
