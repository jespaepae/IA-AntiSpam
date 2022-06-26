from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from nltk.lm.vocabulary import Vocabulary
from nltk import download
import itertools
import string
import numpy as np
from lectura import readEmails, cleanText, read_email
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

vectorizer = TfidfVectorizer()

def entrenar_tf_idf(clean_subjects):
    y = []
    for i in range(11):  #9494 correos de entrenamiento que son spam
        y.append(1)
    for i in range(11): #10828 correos del conjunto de entrenamiento 
        y.append(0)
    y = np.array(y)
    #A continuación, leemos los correos del conjunto de validación y los limpiamos para eliminar símbolos de puntuación y "stopwords".
    spam_subjects_val = readEmails("Enron-Spam-Splited/val/no_deseado/")
    ham_subjects_val = readEmails("Enron-Spam-Splited/val/legítimo/")
    clean_spam_subjects_val = [cleanText(s) for s in spam_subjects_val]
    clean_ham_subjects_val = [cleanText(s) for s in ham_subjects_val]
    clean_subjects_val = np.concatenate((clean_spam_subjects_val, clean_ham_subjects_val))
    print(clean_subjects_val)
    #Tokenizamos las palabras encontradas en el conjunto de validación, y le pedimos al modelo que nos muestre la clasificación realizada de los 50 primeros correos (que deberían ser spam).
    x_val = vectorizer.transform(clean_subjects_val).toarray()
    vectors = vectorizer.fit_transform(clean_subjects)
    print(vectors.toarray())
    knc = KNeighborsClassifier(n_neighbors=1)
    knc.fit(vectors, y)
    print(knc.predict(vectors.toarray()[0:50]))
    y_pred_nb = knc.predict(x_val)     # Vector con la clasificación del filtro
    y_true_nb = []                     # Vector con los resultados que debería dar el filtro
    for i in range(2374):
        y_true_nb.append(1)
    for i in range(2708):
        y_true_nb.append(0)
    y_true_nb = np.array(y_true_nb)
    
    
    cm = confusion_matrix(y_true_nb, y_pred_nb)
    f, ax = plt.subplots(figsize =(6,5))
    sns.heatmap(cm,annot = True,fmt = ".0f",ax=ax,cmap="Blues")
    plt.xlabel("y_pred_nb")
    plt.ylabel("y_true_nb")
    plt.show()