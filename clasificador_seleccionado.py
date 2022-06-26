#Imports
from sklearn.feature_extraction.text import CountVectorizer
from nltk import download
import numpy as np
from lectura import readEmails, cleanText, read_email
from sklearn.naive_bayes import MultinomialNB
from filtro_naive_bayes import entrenar_modelo, clasificador_nb
from filtro_tf_idf import entrenar_tf_idf, clasificador_tf_idf
download('punkt', download_dir='.')
download('stopwords')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
#Code
spam_subjects = readEmails("Enron-Spam-Splited/train/no_deseado/") #9494
ham_subjects = readEmails("Enron-Spam-Splited/train/legítimo/")   #10828
clean_spam_subjects = [cleanText(s) for s in spam_subjects]
number_spams = len(clean_spam_subjects)
clean_ham_subjects = [cleanText(s) for s in ham_subjects]
number_ham = len(clean_ham_subjects)
clean_subjects = np.concatenate((clean_spam_subjects, clean_ham_subjects), axis=0)
vectorizer = CountVectorizer()
clf = MultinomialNB(alpha=1)
x = vectorizer.fit_transform(clean_subjects)
y = []
for i in range(number_spams):  
    y.append(1)
for i in range(number_ham): 
    y.append(0)
y = np.array(y)
clf.fit(x.toarray(), y)
spam_subjects_val = readEmails("Enron-Spam-Splited/val/no_deseado/")
ham_subjects_val = readEmails("Enron-Spam-Splited/val/legítimo/")
clean_spam_subjects_val = [cleanText(s) for s in spam_subjects_val]
clean_ham_subjects_val = [cleanText(s) for s in ham_subjects_val]
clean_subjects_val = np.concatenate((clean_spam_subjects_val, clean_ham_subjects_val))
x_val = vectorizer.transform(clean_subjects_val).toarray()
y_pred_nb = clf.predict(x_val)     
y_true_nb = []                     
for i in range(len(clean_spam_subjects_val)):
    y_true_nb.append(1)
for i in range(len(clean_ham_subjects_val)):
    y_true_nb.append(0)
y_true_nb = np.array(y_true_nb)
cm = confusion_matrix(y_true_nb, y_pred_nb)
f, ax = plt.subplots(figsize =(6,5))
sns.heatmap(cm,annot = True,fmt = ".0f",ax=ax,cmap="Blues")
plt.xlabel("y_pred_nb")
plt.ylabel("y_true_nb")
plt.show()

#evaluable function
def es_mensaje_no_deseado(url):
    subject = read_email(url)
    clean_subject = [cleanText(s) for s in subject]
    x = vectorizer.transform(clean_subject).toarray()
    res = clf.predict(x)
    if res[0] == 0:
        return False
    else:
        return True
print("AZUCENAAAAAAA")
print(es_mensaje_no_deseado("Enron-Spam-Splited/train/no_deseado/0"))