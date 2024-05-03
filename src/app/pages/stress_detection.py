import streamlit as st
import requests
import json
import utils_app

# Configura la página de Streamlit
st.set_page_config(page_title="Inferencia de Estrés con RoBERTa", layout="wide")


def main():
    st.title("Detector de Estrés con RoBERTa")
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
