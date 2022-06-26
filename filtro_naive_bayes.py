#Para representar de forma numérica el contenido de los correos electrónicos usamos como modelo de lenguaje el conocido como bolsa de palabras 
#Bibliografía: https://scikit-learn.org/stable/modules/feature_extraction.html
from sklearn.feature_extraction.text import CountVectorizer
from nltk.lm.vocabulary import Vocabulary
from nltk import download
import itertools
import string
import numpy as np
from lectura import readEmails, cleanText, read_email
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

vectorizer = CountVectorizer()
clf = MultinomialNB(alpha=1)

def entrenar_modelo(clean_subjects, number_spams, number_ham):
    
    #Tokenizamos las palabras que encontramos en los correos y contamos las ocurrencias de cada una, y las representamos en una matriz
    #en la que las filas se correponden con los distintos correos, mientras que las columnas son las distintas palabras que conforman el vocabulario.
    
    x = vectorizer.fit_transform(clean_subjects)
    # Mostramos las 50 primeras palabras del vocabulario, que se correponden con las 50 primeras columnas de la matriz.
    print(vectorizer.get_feature_names_out()[:50])
    #Mostramos las filas y columnas de la matriz en forma de array. Cada elemento representa el número de veces que aparece la palabra de la columna en el correo de la fila.
    print(x.toarray())
    #Creamos un array, en el que los primeros elementos serán 1s, indicándole al clasificador que los correos correspondientes son considerados spam, 
    #y los últimos elementos serán 0s, que se corresponden con correos legítimos.
    y = []
    for i in range(number_spams):  #9494 correos de entrenamiento que son spam
        y.append(1)
    for i in range(number_ham): #10828 correos del conjunto de entrenamiento que no son spams
        y.append(0)
    y = np.array(y)
    #Entrenamos el clasificador de Naive Bayes Multinomial con el conjunto de entrenamiento y mostramos cómo clasificaría los 50 primeros correos del 
    #conjunto de entrenamiento (como son correos no deseados los debería clasificar como spam, es decir, con un 1).
    
    
    clf.fit(x.toarray(), y)
    
    print(clf.predict(x.toarray()[0:50]))
    #A continuación, leemos los correos del conjunto de validación y los limpiamos para eliminar símbolos de puntuación y "stopwords".
    spam_subjects_val = readEmails("Enron-Spam-Splited/val/no_deseado/")
    ham_subjects_val = readEmails("Enron-Spam-Splited/val/legítimo/")
    clean_spam_subjects_val = [cleanText(s) for s in spam_subjects_val]
    clean_ham_subjects_val = [cleanText(s) for s in ham_subjects_val]
    clean_subjects_val = np.concatenate((clean_spam_subjects_val, clean_ham_subjects_val))
    print(clean_subjects_val)
    #Tokenizamos las palabras encontradas en el conjunto de validación, y le pedimos al modelo que nos muestre la clasificación realizada de los 50 primeros correos (que deberían ser spam).
    x_val = vectorizer.transform(clean_subjects_val).toarray()
    print(clf.predict(x_val[:50]))
    #Sin embargo, como podemos comprobar, no todos los elementos son 1s, por lo que el modelo clasifica erróneamente algunos correos.
    #Creamos una matriz de confusión para evaluar el desempeño del filtro, usando la librería seaborn para mostrarla.
    #Bibliografía: https://www.milindsoorya.com/blog/build-a-spam-classifier-in-python https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    y_pred_nb = clf.predict(x_val)     # Vector con la clasificación del filtro
    y_true_nb = []                     # Vector con los resultados que debería dar el filtro
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
#Por tanto, el filtro usando bolsa de palabras como modelo del lenguaje y naive Bayes Multinomial como modelo clasificador 
#es el siguiente (se le debe proporcionar la url donde se encuentra el correo que se quiere clasificar)
def clasificador_nb(url):
    subject = read_email(url)
    clean_subject = [cleanText(s) for s in subject]
    print(subject)
    x = vectorizer.transform(clean_subject).toarray()
    res = clf.predict(x)
    if res[0] == 0:
        return False
    else:
        return True