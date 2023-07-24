# deeptrue
Clasificación de Texto con Modelos de Aprendizaje Automático
Este proyecto es una implementación básica de clasificación de texto utilizando modelos de aprendizaje automático en TensorFlow y Gensim. El objetivo del proyecto es analizar el texto extraído de un documento PDF y clasificarlo en función de la veracidad y el sentimiento.

Requisitos
Antes de ejecutar el código, asegúrate de tener instalados los siguientes paquetes:

Python (versión 3.x)
pandas
numpy
TensorFlow (versión 2.x)
PyPDF2
gensim
Puedes instalar estos paquetes utilizando pip con el siguiente comando:


pip install pandas numpy tensorflow PyPDF2 gensim
Descripción del código
El código consta de varios pasos principales:

Extracción de texto desde un archivo PDF: El código utiliza la librería PyPDF2 para extraer el texto de un documento PDF especificado en la variable ruta_archivo_pdf.

Preprocesamiento de datos: Se ajustan las etiquetas para que coincidan con la cantidad de textos y se crea un DataFrame que contiene los datos de texto, veracidad y sentimiento.

Representación de texto como secuencias numéricas: Se convierte cada texto en una secuencia de números enteros utilizando un diccionario de vocabulario creado a partir de todas las palabras en el conjunto de textos.

Creación de un modelo Word2Vec pre-entrenado: Se carga un modelo Word2Vec pre-entrenado en español utilizando la función load_facebook_model de Gensim. El modelo debe estar en formato binario y se especifica la ruta correcta en la variable ruta_del_modelo_preentrenado.

Combinación de secuencias numéricas con Word2Vec: Se combinan las secuencias numéricas con los vectores densos obtenidos del modelo Word2Vec para crear secuencias de entrada para los modelos de clasificación.

Creación y entrenamiento de los modelos: Se crean dos modelos de clasificación utilizando TensorFlow Keras. Uno para clasificar la veracidad y otro para clasificar el sentimiento. Los modelos se entrenan con los datos de entrada y las etiquetas correspondientes.

Guardado y carga de los modelos entrenados: Los modelos entrenados se guardan en archivos HDF5 para su posterior reutilización.

Clasificación de nueva cadena de texto: Se demuestra cómo utilizar los modelos entrenados para clasificar una nueva cadena de texto proporcionada como ejemplo.

Ejecución del código
Antes de ejecutar el código, asegúrate de tener el archivo PDF 1.El Aura.pdf en el mismo directorio que el script. Además, asegúrate de proporcionar la ruta correcta del modelo Word2Vec pre-entrenado en la variable ruta_del_modelo_preentrenado.

Una vez que los requisitos se cumplan, puedes ejecutar el código desde la terminal o el entorno de desarrollo Python de tu elección.
