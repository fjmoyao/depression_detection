import streamlit as st
import utils_app
# Configure the Streamlit page
st.set_page_config(page_title="Stress Inference with RoBERTa", layout="wide")

def main():
    # Title and introduction of the project
    st.title("Welcome to the Stress Detector with RoBERTa")
    st.write("""
        This application uses an advanced natural language processing model to analyze texts and determine 
        stress levels. Below, you can explore how the model works and test it with your own texts.
    """)

    # Explanation of the project functionalities
    st.header("Functionalities")
    st.write("""
        - **Text Analysis:** Enter any text and the model will determine if the content expresses stress.
        - **Instant Results:** Get results in seconds.
        - **Simple Interface:** An easy-to-use interface that does not require technical knowledge.
    """)

    text = st.text_area("Enter the text for analysis:", height=15)
    
    if st.button("Analyze"):
        if text:
            output = utils_app.query({"inputs": text,})
            result = utils_app.get_label_score(output)

            label = list(result.keys())[0]
            label = str(label).replace("1","Stress").replace("0","Neutral")
            
            st.write("Prediction Result:")
            st.write(label)

        else:
            st.error("Please enter some text for analysis.")

if __name__ == "__main__":
    main()
