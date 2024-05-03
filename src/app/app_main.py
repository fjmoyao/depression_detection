import streamlit as st
import utils_app
# Configure the Streamlit page
st.set_page_config(page_title="Inferencia de Estrés con RoBERTa", layout="wide")

def main():
    # Title and introduction of the project
    st.title("Bienvenido al Detector de Estrés con RoBERTa")
    st.write("""
        Esta aplicación utiliza un modelo avanzado de procesamiento de lenguaje natural para analizar textos y determinar 
        niveles de estrés. A continuación, puedes explorar cómo funciona el modelo y probarlo con tus propios textos.
    """)

    # Explanation of the project functionalities
    st.header("Funcionalidades")
    st.write("""
        - **Análisis de Texto:** Ingrese cualquier texto y el modelo determinará si el contenido expresa estrés.
        - **Resultados Instantáneos:** Obtén resultados en segundos.
        - **Interfaz Sencilla:** Una interfaz fácil de usar que no requiere conocimientos técnicos.
    """)

    text = st.text_area("Ingrese el texto para analizar:", height=15)
    
    if st.button("Analizar"):
        if text:
            output = utils_app.query({"inputs": text,})
            result = utils_app.get_label_score(output)

            label = list(result.keys())[0]
            label = str(label).replace("1","Stress").replace("0","Neutral")
            
            st.write("Resultado de la Predicción:")
            st.write(label)

        else:
            st.error("Por favor, ingrese algún texto para analizar.")


if __name__ == "__main__":
    main()
