# Detección de Trastornos Mentales (Estrés) Utilizando NLP

3197

Este repositorio contiene un conjunto de modelos de aprendizaje automático y técnicas de procesamiento de lenguaje natural para la detección de signos de estrés en textos. El proyecto utiliza modelos avanzados como RoBERTa para clasificar textos que pueden indicar la presencia de estrés en individuos, basándose en sus expresiones escritas.

## Objetivo
El objetivo principal de este proyecto es desarrollar y afinar modelos de NLP capaces de identificar indicativos de trastornos mentales, específicamente estrés, a partir de entradas de texto. Esto puede ser útil para aplicaciones en salud mental, donde la detección temprana puede permitir intervenciones más efectivas.

## Estructura del repositorio
``` 
/
|- data/                  # Carpeta para datasets utilizados y generados durante el proyecto
|- models/                # Modelos entrenados y scripts de entrenamiento
|- notebooks/             # Jupyter notebooks para exploración de datos y resultados
|- src/                   # Código fuente para el entrenamiento y evaluación de modelos
|- requirements.txt       # Dependencias necesarias para replicar el entorno de desarrollo
|- README.md              # Descripción del proyecto, instrucciones de uso y colaboración
``` 

## Instalación

Para instalar y configurar el entorno necesario para ejecutar los modelos y scripts, sigue estos pasos:

1. Clona este repositorio:
2. Instala las dependencias:
   ```
   pdm install
   ```
## Uso 
Para utilizar los modelos para detectar estrés en textos, puedes seguir los ejemplos proporcionados en los Jupyter notebooks dentro de la carpeta notebooks/. Para entrenar un modelo desde cero o utilizar un modelo preentrenado para inferencia, revisa los scripts en la carpeta **src/**.