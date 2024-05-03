
import re
import spacy
import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier
import joblib
import torch
from torch.optim import AdamW
from sklearn.metrics import f1_score

#nlp = spacy.load('en_core_web_sm')
#list_stop_words = stop_words = nlp.Defaults.stop_words

def clean_text(text, keep=None, remove_words=False):
    """
    Limpia el texto eliminando caracteres no deseados, palabras abreviadas opcionales,
    normaliza los espacios y convierte el texto a minúsculas.

    Args:
        text (str): El texto a procesar.
        keep (str, optional): Controla qué caracteres adicionales mantener. 
            'None' para solo letras y espacios, 'sen' para incluir también puntuación básica. Default is None.
        remove_words (bool, optional): Si es True, elimina una lista predefinida de palabras abreviadas.
            Default is False.

    Returns:
        str: El texto procesado y limpio.
    """
    # Lista de palabras a eliminar
    words_to_remove = ["b.s", "p.s", "i.e", "a.k.a", "a.m", "p.m", "e.g"]
    # Crear una expresión regular a partir de la lista
    regex_pattern = r'\b(' + '|'.join(words_to_remove) + r')\b'

    if keep == None:
        text = re.sub(r'[^a-zA-Z\s]', '', text)
    elif keep == "sen":
        text = re.sub(r'[^a-zA-Z\s.?!,]', '', text)

    if remove_words:
        text = re.sub(regex_pattern, '', text)

    # Normalizar espacios y convertir a minúsculas
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.lower()
    return text.strip()

def word_freq(doc):
    frecuencias ={}
    # Contar frecuencias de cada token que sea una palabra
    for token in doc:
        if token.is_alpha:  # Asegurarse de que el token sea alfabético
            word = token.text.lower()  # Convertir a minúsculas para normalizar
            if word in frecuencias:
                frecuencias[word] += 1
            else:
                frecuencias[word] = 1
    #frecuencias =    pd.DataFrame.from_dict(frecuencias, orient="index").reset_index()
    #frecuencias.columns= ["word","count"]
    return frecuencias #.sort_values(ascending=False)

def remove_stopwords(doc, lemma=False, remove_stop=True):
    """
    Procesa un documento de SpaCy para eliminar las palabras vacías y/o aplicar lematización a sus tokens.

    Esta función permite la eliminación opcional de palabras vacías (stopwords) y la lematización de los tokens en un objeto documento de SpaCy. El resultado es una cadena de texto única con los tokens procesados.

    Parámetros:
        doc (spacy.tokens.Doc): Un objeto documento de SpaCy que contiene los tokens a procesar.
        lemma (bool): Si es True, los tokens serán lematizados. El valor predeterminado es False.
        remove_stop (bool): Si es True, las palabras vacías serán eliminadas del documento. El valor predeterminado es True.

    Devoluciones:
        str: Una cadena que contiene el texto procesado basado en las opciones especificadas.

    Ejemplos:
        >>> import spacy
        >>> nlp = spacy.load('en_core_web_sm')  # Asegúrate de tener el modelo cargado adecuadamente
        >>> texto = "This is a sample text with several stopwords."
        >>> doc = nlp(texto)
        >>> print(remove_stopwords(doc, lemma=True, remove_stop=True))
        "sample text several"

        Este ejemplo muestra cómo eliminar palabras vacías y aplicar lematización a un texto en inglés.
    """
    first_person_pronouns = ['i', 'me', 'my', 'mine', "myself"]

    if (remove_stop and lemma):
        clean_txt = [token.lemma_ for token in doc if ((not token.is_stop) | (token.text in first_person_pronouns))]
    elif remove_stop:
        clean_txt = [token.text for token in doc if ((not token.is_stop) | (token.text in first_person_pronouns))]
    elif lemma:
        clean_txt = [token.lemma_ for token in doc]
    else:
        clean_txt = [token.text for token in doc]

    clean_txt = " ".join(clean_txt)
    return clean_txt


def calculate_pronoun_frequency(doc):
    """
    Calcula la frecuencia de los pronombres personales singulares de primera persona en un texto.

    Esta función procesa un texto usando SpaCy para identificar y contar los pronombres personales 
    singulares de primera persona ('i', 'me', 'my', 'mine','myself'). Luego, calcula la frecuencia de estos 
    pronombres dividiendo su cantidad total por el número total de palabras en el texto, excluyendo los 
    signos de puntuación para obtener una medida más precisa.

    Parámetros:
    - texto (str): El texto a analizar.

    Devuelve:
    - float: La frecuencia de los pronombres personales de primera persona, o 0 si no hay palabras.

    Ejemplo:
    >>> doc = nlp("I fell it's all my fault. I hate myself")
    >>> print(calculate_pronoun_frequency(doc))
    0.36363636363636365
    """
    # Define la lista de pronombres personales
    first_person_pronouns = {'i', 'me', 'my', 'mine', "myself"}

    # Cuenta los pronombres
    pronoun_count = sum(
        1 for token in doc if token.lemma_.lower() in first_person_pronouns)

    # Calcula el numero total de palabras
    total_words = len(doc)

    # Evita la division por cero
    if total_words > 0:
        return pronoun_count / total_words
    else:
        return 0


def avg_word_length(doc):
    """
    Calcula la longitud promedio de las palabras en un documento, excluyendo los signos de puntuación.

    Esta función toma un documento SpaCy procesado y computa la longitud promedio de las palabras,
    donde cada 'palabra' es definida como un token que no es un signo de puntuación. Si no hay palabras en el documento
    (es decir, todos los tokens son signos de puntuación), devuelve 0 para evitar la división por cero.

    Parámetros:
    - doc (spacy.tokens.doc.Doc): Un documento procesado por SpaCy.

    Devuelve:
    - float: La longitud promedio de las palabras en el documento, o 0 si el documento no contiene palabras.

    Ejemplo:
    >>> import spacy
    >>> nlp = spacy.load('en_core_web_sm')
    >>> doc = nlp("Hello, world! This is a test.")
    >>> print(avg_word_length(doc))
    3.8
    """
    words = [token.text for token in doc if not token.is_punct]
    return sum(len(word) for word in words) / len(words) if words else 0


def avg_sentence_length(doc):
    """
    Calcula la longitud promedio de las oraciones en un documento.

    Esta función toma un documento SpaCy procesado y computa la longitud promedio de las oraciones,
    donde cada 'oración' es definida por el analizador de oraciones de SpaCy. Si el documento no contiene
    oraciones, la función devuelve 0 para evitar la división por cero.

    Parámetros:
    - doc (spacy.tokens.doc.Doc): Un documento procesado por SpaCy.

    Devuelve:
    - float: La longitud promedio de las oraciones en el documento, o 0 si el documento no contiene oraciones.

    Ejemplo:
    >>> import spacy
    >>> nlp = spacy.load('en_core_web_sm')
    >>> doc = nlp("Hello, world! This is a test. This is another test.")
    >>> print(avg_sentence_length(doc))
    5.0
    """
    sentences = list(doc.sents)
    if len(sentences) == 0:
        return 0  # Evita división por cero si no hay oraciones
    return sum(len(sentence) for sentence in sentences) / len(sentences)


def lexical_diversity(doc):
    """
    Calcula la diversidad léxica de un documento procesado por SpaCy.

    La diversidad léxica se define como la relación entre el número de tipos únicos 
    de palabras y el número total de palabras en el documento, excluyendo signos 
    de puntuación. Esta métrica proporciona una visión cuantitativa de la riqueza 
    del vocabulario utilizado en el documento.

    Parámetros:
    - doc (spacy.tokens.doc.Doc): Un documento procesado por SpaCy. Este documento debe 
      haber sido procesado previamente utilizando el pipeline de SpaCy para garantizar 
      que los tokens y las entidades estén correctamente identificados.

    Devuelve:
    - float: La diversidad léxica del documento, que es la proporción de tipos únicos 
      de palabras sobre el total de palabras. Retorna 0 si el documento no contiene 
      palabras para evitar la división por cero.

    Ejemplo:
    >>> import spacy
    >>> nlp = spacy.load('en_core_web_sm')
    >>> doc = nlp("Hello, world! Hello, everyone.")
    >>> print(lexical_diversity(doc))
    0.5
    """
    words = [
        token.text for token in doc if not token.is_punct]  # Filtra los signos de puntuación
    types = set(words)  # Conjunto de tipos únicos de palabras
    if len(words) == 0:
        return 0  # Retorna 0 si no hay palabras para evitar división por cero
    return len(types) / len(words)  # Calcula la relación Tipo-Token


def average_adv_adj(doc, kind=None):
    """
    Calcula el promedio de adverbios y adjetivos por oración en un documento procesado por SpaCy.

    Esta función permite especificar si se desea calcular el promedio de solo adverbios, solo adjetivos,
    o ambos. Esto se hace mediante el parámetro 'kind'. Analiza el documento y cuenta los adverbios y
    adjetivos, luego divide este conteo por el número total de oraciones para obtener el promedio deseado.

    Parámetros:
    - doc (spacy.tokens.doc.Doc): El documento procesado por SpaCy.
    - kind (str, opcional): Especifica el tipo de palabra para calcular el promedio. Puede ser 'adj' para
      adjetivos, 'adv' para adverbios, o None para ambos. El valor predeterminado es None.

    Devuelve:
    - float: El promedio de adverbios, adjetivos o ambos por oración. Retorna 0 si no hay oraciones.

    Ejemplos:
    >>> doc = nlp("The quick brown fox jumps over the lazy dog. It runs very quickly.")
    >>> print(average_adv_adj(doc))  # Promedio combinado
    1.0
    >>> print(average_adv_adj(doc, kind='adj'))  # Solo adjetivos
    0.5
    >>> print(average_adv_adj(doc, kind='adv'))  # Solo adverbios
    0.5
    """

    adv_count = 0
    adj_count = 0
    total_sentences = len(list(doc.sents))

    # Contar adverbios y adjetivos en el documento
    for token in doc:
        if token.pos_ == 'ADV':
            adv_count += 1
        elif token.pos_ == 'ADJ':
            adj_count += 1

    # Evitar división por cero si no hay oraciones
    if total_sentences == 0:
        return 0

    # Calcular el promedio según el tipo especificado
    if kind == "adj":
        return adj_count / total_sentences if total_sentences > 0 else 0
    elif kind == "adv":
        return adv_count / total_sentences if total_sentences > 0 else 0
    else:
        return (adv_count + adj_count) / total_sentences if total_sentences > 0 else 0


def average_passive_sentences(doc):
    """
    Cuenta el número de oraciones pasivas en un documento procesado por SpaCy.

    Una oración se considera pasiva si contiene un auxiliar de voz pasiva ('auxpass')
    o un sujeto pasivo ('nsubjpass') asociado con un verbo. La función recorre cada 
    oración y cada token dentro de la oración para identificar estas estructuras.

    Parámetros:
    - doc (spacy.tokens.doc.Doc): Documento procesado por SpaCy que será analizado. 

    Devuelve:
    - float: Promedio de oraciones pasivas encontradas en el documento. Si no se encuentran
      oraciones pasivas, devuelve 0.

    Ejemplo:
    >>> text = "The ball was thrown by the player. She makes a great throw."
    >>> doc = nlp(text)
    >>> print(average_passive_sentences(doc))
    1
    """

    passive_count = 0
    total_sentences = 0

    for sent in doc.sents:
        total_sentences += 1
        for token in sent:
            # Buscar estructuras de voz pasiva
            if token.dep_ == 'auxpass' or (token.dep_ == 'nsubjpass' and token.head.pos_ == 'VERB'):
                passive_count += 1
                break  # Solo cuenta una vez por oración

    return passive_count / total_sentences


def get_models():
    """
    Inicializa un diccionario de modelos de aprendizaje automático con parámetros predefinidos.

    Esta función crea y configura varios modelos populares de aprendizaje automático, cada uno con
    configuraciones iniciales específicas. Los modelos se crean utilizando las bibliotecas scikit-learn
    y XGBoost, y se ajustan mediante pipelines donde es necesario para estandarizar los datos.

    Devuelve:
        dict: Un diccionario que contiene los modelos inicializados con sus nombres respectivos.

    Los modelos incluidos son:
    - Regresión Logística
    - K-Vecinos Más Cercanos
    - Árbol de Decisión
    - SVM
    - Naive Bayes
    - XGBoost
    - Gradient Boosting
    - Random Forest
    - AdaBoost

    """
    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(solver='saga', C=70.0)),
        'K-Nearest Neighbors': make_pipeline(StandardScaler(), KNeighborsClassifier()),
        'Decision Tree': DecisionTreeClassifier(max_depth=1),
        'SVM': make_pipeline(StandardScaler(), SVC()),
        'Naive Bayes': make_pipeline(StandardScaler(), GaussianNB()),
        'XGBoost': XGBClassifier(n_estimators=11, max_depth=1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=10),
        'Random Forest': RandomForestClassifier(n_estimators=10),
        'AdaBoost': AdaBoostClassifier(n_estimators=12)
    }
    return models


def evaluate_model(model, X, y, model_name=None, model_ext=None):
    """
    Evalúa un modelo de aprendizaje automático utilizando validación cruzada.

    Parámetros:
        model (sklearn.base.BaseEstimator): Modelo de aprendizaje automático a evaluar.
        X (array-like): Conjunto de datos de características utilizadas para la evaluación del modelo.
        y (array-like): Variable objetivo correspondiente a X.
        model_name (str, opcional): Nombre base para el archivo al guardar el modelo entrenado.
        model_ext (str, opcional): Extensión adicional para el nombre del archivo al guardar el modelo.

    Devoluciones:
        np.array: Puntuaciones F1 obtenidas de la validación cruzada.

    Guarda el modelo en un archivo si `model_name` y `model_ext` son proporcionados. El archivo
    se guardará en el directorio 'data/gold' del directorio padre del directorio de trabajo actual.
    """

    # Define la ruta para guardar el modelo
    data_path = Path(os.getcwd()).parent / "models"
    # Configura los parámetros para la validación cruzada
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=1)
    # Realiza la validación cruzada y calcula las puntuaciones F1
    scores = cross_val_score(model, X, y, scoring='f1',
                             cv=cv, n_jobs=-1, error_score='raise')

    # Guarda el modelo si se proporcionan los nombres
    if model_name is not None and model_ext is not None:
        extName = f"{model_name}_{model_ext}.pkl"
        save_dir = data_path / extName
        joblib.dump(model, save_dir)

    return scores



def train_and_evaluate(model, train_loader, val_loader, device, num_epochs=5):
    optimizer = AdamW(model.parameters(), lr=1e-5)
    best_f1 = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)

        # Evaluación
        model.eval()
        val_true, val_preds = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                val_true.extend(labels.cpu().numpy())
                val_preds.extend(predictions.cpu().numpy())

        val_f1 = f1_score(val_true, val_preds, average='binary')
        if val_f1 > best_f1:
            best_f1 = val_f1  # Actualizar el mejor F1 score visto hasta ahora

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation F1: {val_f1:.4f}')

    return best_f1


def save_model_and_tokenizer(model, tokenizer, save_directory):
    """
    Save the fine-tuned model and tokenizer to a specified directory.

    Args:
        model: The trained RoBERTa model (e.g., an instance of RobertaForSequenceClassification).
        tokenizer: The tokenizer used with the RoBERTa model.
        save_directory (str): The path to the directory where the model and tokenizer will be saved.

    Returns:
        None
    """
    # Create the directory if it does not exist
    import os
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save the model
    model.save_pretrained(save_directory)
    
    # Save the tokenizer associated with the model
    tokenizer.save_pretrained(save_directory)

    print(f"Model and tokenizer have been saved to {save_directory}") 


def prepare_text_for_inference(text, tokenizer, max_length=128):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # Agrega los tokens especiales para RoBERTa '[CLS]' y '[SEP]'
        max_length=max_length,    # Trunca o rellena el texto hasta la longitud máxima
        padding='max_length',     # Rellena hasta `max_length`
        truncation=True,          # Trunca a `max_length` si el texto es más largo
        return_attention_mask=True,
        return_tensors='pt',      # Retorna tensores de PyTorch
    )
    return encoding['input_ids'], encoding['attention_mask']


def make_prediction(text, model, tokenizer, device):
    model.eval()  # Pone el modelo en modo evaluación

    input_ids, attention_mask = prepare_text_for_inference(text, tokenizer)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():  # Desactiva el cálculo de gradientes para inferencia
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1)

    return probabilities.cpu().numpy(), predicted_class.cpu().numpy()