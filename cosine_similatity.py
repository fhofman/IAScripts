from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

frases = [
    "El gato duerme en el sofá",
    "El felino está durmiendo en el sillón",
    "El sillon esta roto por el gato",
    "El gato sirve para cambiar la rueda del auto",
    "Uso el sillón para mirar tele",
]

vectorizador = TfidfVectorizer()
X = vectorizador.fit_transform(frases)   # matriz: una fila por frase

sim = cosine_similarity(X)               # matriz 4x4 de similitudes

for i in range(len(frases)):
    for j in range(i + 1, len(frases)):
        print(f"{sim[i, j]:.2f}  |  {frases[i]}  <->  {frases[j]}")
