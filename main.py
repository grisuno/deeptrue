import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import PyPDF2
import gensim
from gensim.models import Word2Vec
from gensim.models.fasttext import load_facebook_model

# Leer el texto desde el documento PDF
ruta_archivo_pdf = "1.El Aura.pdf"

def leer_texto_desde_pdf(ruta_archivo):
    with open(ruta_archivo, 'rb') as archivo_pdf:
        lector_pdf = PyPDF2.PdfReader(archivo_pdf)
        num_paginas = len(lector_pdf.pages)
        textos = [lector_pdf.pages[pagina].extract_text() for pagina in range(num_paginas)]
    return textos


# Datos ficticios de ejemplo
textos = leer_texto_desde_pdf(ruta_archivo_pdf)

# Ajustar las etiquetas para que coincidan con la cantidad de textos
etiquetas_veracidad = [1] * len(textos)  # Por ejemplo, asignar todas las etiquetas como 1
etiquetas_sentimiento = [0.5] * len(textos)  # Por ejemplo, asignar todas las etiquetas como 0.5

# Crear un diccionario con los datos
data = {
    'Texto': textos,
    'Veracidad': etiquetas_veracidad,
    'Sentimiento': etiquetas_sentimiento
}

# Crear el DataFrame
df = pd.DataFrame(data)

# Convertir texto a secuencias numéricas
vocabulario = set(" ".join(textos).split())
vocabulario = {palabra: indice + 1 for indice, palabra in enumerate(vocabulario)}
secuencias_numericas = [[vocabulario[palabra] for palabra in texto.split()] for texto in textos]

# Pad secuencias a la misma longitud
max_longitud = max(len(seq) for seq in secuencias_numericas)
secuencias_numericas = tf.keras.preprocessing.sequence.pad_sequences(secuencias_numericas, maxlen=max_longitud)

# Convertir etiquetas a matrices one-hot y binarizar etiquetas de sentimiento
etiquetas_veracidad = np.array(etiquetas_veracidad)
etiquetas_veracidad_one_hot = tf.keras.utils.to_categorical(etiquetas_veracidad)
etiquetas_sentimiento_binarias = [1 if sentimiento >= 0.5 else 0 for sentimiento in etiquetas_sentimiento]

# Cargar modelo de fastText pre-entrenado
ruta_del_modelo_preentrenado = "cc.es.300.bin"  # Reemplaza esto con la ruta correcta del modelo
modelo_word2vec = load_facebook_model(ruta_del_modelo_preentrenado)

# Representar palabras del texto en vectores densos usando Word2Vec
secuencias_word2vec = [[modelo_word2vec[palabra] for palabra in texto.split()] for texto in textos]

# Combinar las secuencias de Word2Vec con las secuencias numéricas originales
secuencias_combinadas = np.concatenate([secuencias_numericas, secuencias_word2vec], axis=1)

# Crear modelo de veracidad y análisis de sentimiento y entrenarlos
modelo_veracidad = tf.keras.Sequential()
# ... Definición del modelo de veracidad ...
modelo_veracidad.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo_veracidad.fit(secuencias_combinadas, etiquetas_veracidad_one_hot, epochs=10, batch_size=2)
modelo_sentimiento = tf.keras.Sequential()
modelo_sentimiento.add(layers.Embedding(len(vocabulario) + 1, 100, input_length=max_longitud))
modelo_sentimiento.add(layers.Conv1D(64, 5, activation='relu'))
modelo_sentimiento.add(layers.GlobalMaxPooling1D())
modelo_sentimiento.add(layers.Dense(32, activation='relu'))
modelo_sentimiento.add(layers.Dense(1, activation='sigmoid'))

modelo_sentimiento.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo_sentimiento.fit(secuencias_combinadas, etiquetas_sentimiento_binarias, epochs=10, batch_size=2)

# Guardar los modelos en archivos
modelo_veracidad.save("modelo_veracidad.h5")
modelo_sentimiento.save("modelo_sentimiento.h5")

# Cargar los modelos desde los archivos
modelo_veracidad_cargado = tf.keras.models.load_model("modelo_veracidad.h5")
modelo_sentimiento_cargado = tf.keras.models.load_model("modelo_sentimiento.h5")

# Nueva cadena de texto para verificar
nueva_cadena_texto = "Una nueva investigación demuestra..."
nueva_secuencia_numerica = [vocabulario[palabra] for palabra in nueva_cadena_texto.split()]
nueva_secuencia_numerica = tf.keras.preprocessing.sequence.pad_sequences([nueva_secuencia_numerica], maxlen=max_longitud)

# Representar palabras del texto en vectores densos usando Word2Vec
nueva_secuencia_word2vec = [modelo_word2vec[palabra] for palabra in nueva_cadena_texto.split()]

# Combinar las secuencias de Word2Vec con la secuencia numérica original
nueva_secuencia_combinada = np.concatenate([nueva_secuencia_numerica, nueva_secuencia_word2vec], axis=1)

# Predicciones de veracidad y sentimiento
prediccion_veracidad = modelo_veracidad_cargado.predict(nueva_secuencia_combinada)
prediccion_sentimiento = modelo_sentimiento_cargado.predict(nueva_secuencia_combinada)[0][0]

# Resultado de veracidad (0 o 1) y sentimiento (entre 0 y 1)
resultado_veracidad = np.argmax(prediccion_veracidad)
resultado_sentimiento = 1 if prediccion_sentimiento >= 0.5 else 0

print("Longitud de textos:", len(textos))
print("Longitud de etiquetas_veracidad:", len(etiquetas_veracidad))
print("Longitud de etiquetas_sentimiento:", len(etiquetas_sentimiento))

print("Resultado de veracidad:", resultado_veracidad)
print("Resultado de sentimiento:", prediccion_sentimiento)
