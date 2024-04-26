
import re
import spacy 
import pandas as pd
import os 
from pathlib import Path


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
