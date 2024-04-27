
import re
import spacy 
import pandas as pd
import os 
from pathlib import Path


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
    first_person_pronouns = {'i', 'me', 'my', 'mine',"myself"}
    
    # Cuenta los pronombres
    pronoun_count = sum(1 for token in doc if token.lemma_.lower() in first_person_pronouns)
    
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
    words = [token.text for token in doc if not token.is_punct]  # Filtra los signos de puntuación
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
    total_sentences =0

    for sent in doc.sents:
        total_sentences+=1
        for token in sent:
            # Buscar estructuras de voz pasiva
            if token.dep_ == 'auxpass' or (token.dep_ == 'nsubjpass' and token.head.pos_ == 'VERB'):
                passive_count += 1
                break  # Solo cuenta una vez por oración

    return passive_count/ total_sentences


