import streamlit as st
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
nltk.download('punkt')
nltk.download('stopwords')
chatbot=pipeline("question-answering", model="deepset/bert-base-cased-squad2")
def preprocess_input(user_input):
    stop_words=set(stopwords.words('english'))
    words=word_tokenize(user_input)
    filtered_words=[word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)
def healthcare_chatbot(user_input):
    user_input=preprocess_input(user_input).lower()
    if "sneeze" in user_input or "sneezing" in user_input:
        return "Frequent sneezing may indicate allergies or a cold. Please Consult a doctor if symptoms persist."
    elif "symptom" in user_input:
        return "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice."
    elif "appointment" in user_input:
        return "Would you like me to schedule an appointment with a doctor?"
    elif "medication" in user_input:
        return "It's important to take your prescribed medications regularly. If you have concerns, Consult your doctor."
    else:
        context= """
        Common healthcare-related scenarios include symptoms of colds, flu and allergies,
        along with medication guidance and appointment scheduling.
        """
        response=chatbot(question=user_input, context=context)
        return response['answer']
def main():
    st.markdown("""
    <style>
        .chat-container {
            max-width: 600px;
            margin: auto;
        }
        .chat-bubble {
            padding: 10px;
            border-radius: 10px;
            margin: 5px 0;
        }
        .user-bubble {
            background-color: #0084ff;
            color: white;
            text-align: right;
        }
        .assistant-bubble {
            background-color: #e0e0e0;
            color: black;
            text-align: left;
        }
    </style>
""", unsafe_allow_html=True)
    # Sidebar
    with st.sidebar:
        st.title("AI Health Assistant")
        st.write("🔹 Get instant health advice powered by AI.")
        st.write("🔹 Type your symptoms and receive suggestions.")

    # Main Chat Interface
    st.markdown("<h1 style='text-align: center;'>Healthcare Assistant 🤖</h1>", unsafe_allow_html=True)
    st.write("---")
    
    user_input = st.text_input("Describe your symptoms:", "", key="user_input")
    
    if st.button("Submit"):
        if user_input:
            st.markdown(f"<div class='chat-container'><div class='chat-bubble user-bubble'>User: {user_input}</div></div>", unsafe_allow_html=True)
            
            with st.spinner("Thinking..."):
                time.sleep(2)  # Simulate processing time
                response = healthcare_chatbot(user_input)

            st.markdown(f"<div class='chat-container'><div class='chat-bubble assistant-bubble'>Health Assistant: {response}</div></div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a symptom to proceed.")
    
if __name__=="__main__":
    main()


