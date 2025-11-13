import streamlit as st
import pandas as pd
import numpy as np
import spacy
import tensorflow as tf
import matplotlib.pyplot as plt

# Scikit-learn imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Helper Functions (Caching Models) ---

@st.cache_resource
def load_mnist_model():
    """Loads the pre-trained MNIST CNN model."""
    try:
        model = tf.keras.models.load_model('mnist_cnn.keras')
        return model
    except Exception as e:
        st.error(f"Error loading MNIST model: {e}")
        st.error("Have you run 'python train_mnist.py' first to create the 'mnist_cnn.keras' file?")
        return None

@st.cache_resource
def load_spacy_model():
    """Loads the spaCy model."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("spaCy 'en_core_web_sm' model not found. Please run:")
        st.code("python -m spacy download en_core_web_sm")
        return None

# --- Main App ---
st.set_page_config(layout="wide")
st.sidebar.title("AI for SE Projects")
page = st.sidebar.selectbox("Choose a project:", 
    ["Home", "Task 1: Iris Classifier", "Task 2: MNIST CNN", "Task 3: NLP with spaCy"])

# --- Home Page ---
if page == "Home":
    st.title("AI for Software Engineering: Project Showcase")
    st.markdown("""
    Welcome to this multi-project app, deployed using Streamlit.
    
    This application demonstrates three different pillars of AI:
    1.  **Classical ML:** A Decision Tree for Iris species classification.
    2.  **Deep Learning:** A CNN for handwritten digit recognition.
    3.  **NLP:** Entity Recognition and Sentiment Analysis on product reviews.
    
    Use the sidebar to navigate to each project.
    """)

# --- Task 1: Iris Classifier ---
elif page == "Task 1: Iris Classifier":
    st.header("Task 1: Classical ML with Scikit-learn")
    st.subheader("Predicting Iris Species with a Decision Tree")

    if st.button("Train and Evaluate Model", key="iris_button"):
        with st.spinner("Training model..."):
            # 1. Load Data
            iris = load_iris()
            X = iris.data
            y = iris.target
            target_names = iris.target_names
            
            # 2. Preprocessing & Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 3. Train
            clf = DecisionTreeClassifier(random_state=42, max_depth=3)
            clf.fit(X_train, y_train)
            
            # 4. Evaluate
            y_pred = clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            st.success(f"Model Trained! Accuracy: {accuracy:.4f}")
            st.subheader("Classification Report")
            st.dataframe(report_df)

# --- Task 2: MNIST CNN ---
elif page == "Task 2: MNIST CNN":
    st.header("Task 2: Deep Learning with TensorFlow")
    st.subheader("Classifying Handwritten Digits with a CNN")
    
    model = load_mnist_model()
    
    if model:
        st.success("Pre-trained MNIST model loaded successfully.")
        
        if st.button("Show 5 Sample Predictions", key="mnist_button"):
            # Load test data
            (_, _), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
            X_test_norm = X_test.reshape((-1, 28, 28, 1)) / 255.0

            # Get predictions for the first 5 images
            predictions = model.predict(X_test_norm[:5])
            
            fig, axes = plt.subplots(1, 5, figsize=(12, 3))
            for i in range(5):
                ax = axes[i]
                ax.imshow(X_test[i], cmap='gray')
                pred_label = np.argmax(predictions[i])
                true_label = y_test[i]
                
                color = 'green' if pred_label == true_label else 'red'
                ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}", color=color)
                ax.axis('off')
            
            st.pyplot(fig)

# --- Task 3: NLP with spaCy ---
elif page == "Task 3: NLP with spaCy":
    st.header("Task 3: NLP with spaCy")
    st.subheader("Amazon Review Analysis (NER & Sentiment)")
    
    nlp = load_spacy_model()
    
    if nlp:
        st.success("spaCy model 'en_core_web_sm' loaded successfully.")
        
        # Sample Amazon-style Reviews
        reviews = [
            "I absolutely love my new Sony WH-1000XM4 headphones! The noise cancellation is amazing.",
            "The battery life on this iPhone 14 is terrible. I regret buying it.",
            "Shipping was fast, but the screen of the Samsung Galaxy Tab arrived cracked. Very disappointed.",
            "Great value for money. The Kindle Paperwhite is the best e-reader I have used.",
            "It's okay. The Logitech mouse works, but the scroll wheel feels cheap."
        ]
        
        # Simple sentiment lexicon
        positive_words = {"love", "amazing", "great", "best", "fast", "value", "good", "excellent"}
        negative_words = {"terrible", "regret", "cracked", "disappointed", "cheap", "bad", "slow", "hate"}
        
        for i, text in enumerate(reviews):
            st.markdown(f"--- \n**Review #{i+1}:** *\"{text}\"*")
            doc = nlp(text)
            
            # --- NER ---
            st.markdown("**Extracted Entities:**")
            entities_found = False
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "GPE", "PERSON"]:
                    st.markdown(f"- `{ent.text}` ({ent.label_})")
                    entities_found = True
            if not entities_found:
                st.markdown("- *None found.*")

            # --- Sentiment ---
            score = 0
            for token in doc:
                if token.lemma_.lower() in positive_words: score += 1
                if token.lemma_.lower() in negative_words: score -= 1
            
            sentiment = "POSITIVE" if score > 0 else ("NEGATIVE" if score < 0 else "NEUTRAL")
            st.markdown(f"**Sentiment:** {sentiment} (Score: {score})")
