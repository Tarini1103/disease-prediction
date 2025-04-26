import pandas as pd
import seaborn as sns
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import re
import joblib
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter
from time import time
import plotly.graph_objects as go
from sklearn.inspection import permutation_importance

# Set page config
st.set_page_config(page_title="Disease Predictor Pro", layout="wide", page_icon="🏥")

@st.cache_data
def load_and_preprocess():
    """Load dataset and convert symptoms to keywords with enhanced preprocessing"""
    try:
        df = pd.read_csv('Symptom2Disease.csv')
        
        # Enhanced stopwords list
        stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
            "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
            'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
            "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
            'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
            'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
            "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
            "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
            "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'ive', 'im',
            'get', 'got', 'like', 'feel', 'feeling', 'still', 'even', 'one', 'two', 'three',
            'also', 'much', 'many', 'lot', 'really', 'maybe', 'perhaps', 'sometimes', 'usually',
            'often', 'always', 'never', 'every', 'since', 'ago', 'would', 'could', 'may', 'might',
            'must', 'shall', 'thats', 'theres', 'whens', 'wheres', 'whys', 'hows', 'ok', 'okay',
            'yes', 'no', 'hi', 'hello', 'hey', 'please', 'thanks', 'thank', 'sorry', 'oh', 'ah',
            'um', 'uh', 'er', 'well', 'actually', 'basically', 'seriously', 'literally', 'right',
            'left', 'back', 'front', 'top', 'bottom', 'side', 'sides', 'part', 'parts', 'area',
            'areas', 'thing', 'things', 'stuff', 'kind', 'kinds', 'sort', 'sorts', 'bit', 'bits',
            'little', 'big', 'small', 'large', 'long', 'short', 'high', 'low', 'good', 'bad',
            'better', 'worse', 'worst', 'best', 'new', 'old', 'young', 'first', 'last', 'next',
            'previous', 'early', 'late', 'recent', 'recently', 'already', 'yet', 'almost', 'nearly',
            'quite', 'rather', 'somewhat', 'too', 'very', 'extremely', 'absolutely', 'completely',
            'totally', 'utterly', 'perfectly', 'exactly', 'precisely', 'certainly', 'definitely',
            'probably', 'possibly', 'likely', 'unlikely', 'usually', 'normally', 'typically',
            'generally', 'especially', 'particularly', 'mainly', 'mostly', 'chiefly', 'primarily',
            'usually', 'commonly', 'frequently', 'regularly', 'often', 'sometimes', 'occasionally',
            'rarely', 'seldom', 'hardly', 'never', 'always', 'constantly', 'continuously'
        }
        
        # Enhanced keyword extraction
        def extract_keywords(text):
            # Remove special chars, numbers, and extra spaces
            text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split and filter words
            words = text.split()
            
            # Remove stopwords and short words
            filtered_words = [
                word for word in words 
                if word not in stop_words 
                and len(word) > 2
                and not word.isnumeric()
            ]
            
            # Remove common verb forms and pronouns
            verb_forms = {'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
            filtered_words = [word for word in filtered_words if word not in verb_forms]
            
            return ' '.join(filtered_words)
        
        df['keyword_symptoms'] = df['text'].apply(extract_keywords)
        
        # Remove empty or very short symptom descriptions
        df = df[df['keyword_symptoms'].str.len() > 10]
        
        # Remove rare diseases (require at least 10 samples per disease)
        disease_counts = df['label'].value_counts()
        valid_diseases = disease_counts[disease_counts >= 10].index
        df = df[df['label'].isin(valid_diseases)]
        
        return df
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error("Please ensure 'Symptom2Disease.csv' is in your directory")
        return None

def train_models(df):
    """Train and save ML models with enhanced parameters"""
    try:
        # Encode diseases
        le = LabelEncoder()
        y = le.fit_transform(df['label'])
        
        # Enhanced TF-IDF Vectorizer
        tfidf = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 3),
            min_df=5,
            max_df=0.8,
            stop_words='english',
            sublinear_tf=True
        )
        
        # Split data with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            df['keyword_symptoms'], y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )
        
        # Transform
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        # Initialize models
        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced_subsample',
                random_state=42,
                n_jobs=-1
            ),
            "Support Vector Machine": SVC(
                C=1.0,
                kernel='linear',
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            "Logistic Regression": LogisticRegression(
                C=0.5,
                solver='lbfgs',
                max_iter=1000,
                multi_class='multinomial',
                class_weight='balanced',
                random_state=42
            )
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            start_time = time()
            model.fit(X_train_tfidf, y_train)
            train_time = time() - start_time
            
            # Evaluate
            y_pred = model.predict(X_test_tfidf)
            y_proba = model.predict_proba(X_test_tfidf) if hasattr(model, "predict_proba") else None
            
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            
            results[name] = {
                "model": model,
                "accuracy": accuracy,
                "report": report,
                "train_time": train_time,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "y_test": y_test  # Added this line to store y_test for confusion matrix
            }
        
        # Save components
        joblib.dump(le, 'label_encoder.pkl')
        joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
        
        return results
        
    except Exception as e:
        st.error(f"Model training failed: {str(e)}")
        return None

def show_disease_info(disease_name):
    """Display additional information about the disease"""
    disease_info = {
        "Psoriasis": {
            "symptoms": "Red patches with silvery scales, dry/cracked skin, itching/burning/soreness",
            "actions": "See a dermatologist, avoid triggers (stress, infections, skin injuries)",
            "treatments": "Topical treatments, light therapy, systemic medications"
        },
        "Migraine": {
            "symptoms": "Throbbing headache, nausea, sensitivity to light/sound",
            "actions": "Rest in quiet/dark room, stay hydrated, identify triggers",
            "treatments": "Pain-relief medications, preventive medications, lifestyle changes"
        },
        "Allergy": {
            "symptoms": "Sneezing, runny nose, itchy eyes/nose, rash/hives",
            "actions": "Avoid allergens, use air filters, keep windows closed during high pollen",
            "treatments": "Antihistamines, decongestants, nasal sprays, immunotherapy"
        },
        "Typhoid": {
            "symptoms": "High fever, headache, stomach pain, weakness",
            "actions": "Seek medical attention, maintain hydration, practice good hygiene",
            "treatments": "Antibiotics, fluid replacement, rest"
        },
        "Dengue": {
            "symptoms": "High fever, severe headache, pain behind eyes, joint/muscle pain",
            "actions": "Rest, stay hydrated, avoid mosquito bites, seek medical care",
            "treatments": "Pain relievers, fluid replacement, hospitalization in severe cases"
        },
        "COVID-19": {
            "symptoms": "Fever, cough, shortness of breath, loss of taste/smell, fatigue",
            "actions": "Isolate, get tested, wear a mask, monitor symptoms, seek care if needed",
            "treatments": "Supportive care, antivirals, oxygen therapy in severe cases"
        },
        "Pneumonia": {
            "symptoms": "Cough with phlegm, fever, chest pain, shortness of breath",
            "actions": "Seek medical care, rest, stay hydrated, monitor breathing",
            "treatments": "Antibiotics or antivirals (depending on cause), oxygen therapy, rest"
        },
        "Common Cold": {
            "symptoms": "Sneezing, runny nose, sore throat, cough, mild fever",
            "actions": "Rest, drink plenty of fluids, avoid cold exposure",
            "treatments": "Over-the-counter cold medications, steam inhalation, warm fluids"
        },
        "Flu (Influenza)": {
            "symptoms": "High fever, chills, muscle aches, fatigue, cough, sore throat",
            "actions": "Rest, stay home, avoid contact with others, stay hydrated",
            "treatments": "Antiviral drugs (if severe), pain relievers, hydration"
        }, 
        "Acid Reflux (GERD)": {
            "symptoms": "Heartburn, regurgitation, chest discomfort after eating",
            "actions": "Avoid spicy/greasy food, eat smaller meals, avoid lying down after eating",
            "treatments": "Antacids, proton pump inhibitors, dietary changes"
        },
        "allergy": {
            "symptoms": "Sneezing, itchy eyes/nose/skin, runny or blocked nose, rashes, swelling, difficulty breathing (in severe cases)",
            "actions": "Identify and avoid allergens, use air purifiers, consult a doctor for testing and advice",
            "treatments": "Antihistamines, decongestants, corticosteroid creams, epinephrine injection (for severe reactions)"
        },
        "Hypertension": {
            "symptoms": "includes headaches, blurred vision, or dizziness in severe cases",
            "actions": "Regularly check blood pressure, reduce salt and processed food intake, maintain a healthy weight, exercise, avoid smoking and alcohol",
            "treatments": "Long-term lifestyle changes, antihypertensive medications as prescribed, routine monitoring by a healthcare provider"
        },
        "Malaria": {
            "symptoms": "Fever, chills, sweating, headache, nausea, vomiting, muscle pain, fatigue",
            "actions": "Seek immediate medical attention, avoid mosquito bites, use mosquito nets and repellents",
            "treatments": "Antimalarial medications (e.g., artemisinin-based combination therapy), supportive care, hydration"
        },
        "diabetes": {
            "symptoms": "Increased thirst, frequent urination, fatigue, blurred vision, slow healing of wounds, unexplained weight loss",
            "actions": "Monitor blood sugar levels, follow a healthy diet, exercise regularly, maintain a healthy weight, consult a doctor",
            "treatments": "Lifestyle changes, oral medications (e.g., metformin), insulin therapy (if needed), regular blood sugar monitoring"
        }


        # Add more diseases as needed
    }
    
    info = disease_info.get(disease_name, {
        "symptoms": "Common symptoms would be listed here",
        "actions": "Consult a healthcare professional for proper diagnosis",
        "treatments": "Treatment varies based on severity and individual factors"
    })
    
    st.markdown(f"""
    **About {disease_name}:**
    - **Common symptoms:** {info['symptoms']}
    - **Recommended actions:** {info['actions']}
    - **Typical treatments:** {info['treatments']}
    """)

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(cm, 
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=classes,
                    y=classes,
                    color_continuous_scale='Blues')
    fig.update_layout(title="Confusion Matrix")
    return fig

def plot_feature_importance(model, feature_names, n=20):
    """Plot feature importance for tree-based models"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-n:][::-1]
        
        fig = px.bar(x=feature_names[indices],
                    y=importances[indices],
                    labels={'x': 'Features', 'y': 'Importance'},
                    title="Top Feature Importances")
        return fig
    return None

def validate_input(text):
    """Validate user input"""
    if not text.strip():
        return False, "Please enter symptoms before predicting"
    if len(text) < 10:
        return False, "Please provide more detailed symptoms (at least 10 characters)"
    if len(text.split()) < 3:
        return False, "Please describe at least 3 symptoms"
    return True, ""

def main():
    st.title("🏥 Advanced Disease Prediction from Symptoms")
    st.markdown("""
    Describe your symptoms in natural language to get possible disease predictions.
    *For example: "I have persistent headache, fever, and fatigue for 3 days"*
    """)
    
    # Initialize session state
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    # Load data
    df = load_and_preprocess()
    if df is None:
        return
    
    # Model training section
    if not st.session_state.models_trained:
        st.subheader("Model Training")
        st.write("The predictive models need to be trained before making predictions.")
        
        if st.button("Train Models", type="primary"):
            with st.spinner("Training models (this may take a few minutes)..."):
                results = train_models(df)
                if results is not None:
                    st.session_state.models_trained = True
                    st.session_state.results = results
                    
                    # Show model comparison
                    st.subheader("Model Comparison")
                    
                    # Create comparison dataframe
                    comparison_data = []
                    for name, result in results.items():
                        comparison_data.append({
                            "Model": name,
                            "Accuracy": result["accuracy"],
                            "Training Time (s)": result["train_time"]
                        })
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    
                    # Display comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.dataframe(comparison_df.style.format({
                            "Accuracy": "{:.2%}",
                            "Training Time (s)": "{:.2f}"
                        }))
                    
                    with col2:
                        fig = px.bar(comparison_df, 
                                    x="Model", 
                                    y="Accuracy",
                                    title="Model Accuracy Comparison",
                                    color="Model")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.rerun()
    
    # Prediction section
    if st.session_state.models_trained:
        try:
            # Load model components
            le = joblib.load('label_encoder.pkl')
            tfidf = joblib.load('tfidf_vectorizer.pkl')
            results = st.session_state.results
            
            # User input
            st.subheader("Symptom Description")
            user_input = st.text_area(
                "Describe your symptoms in detail:",
                "I cant stop sneezing and have really runny nose",
                height=150
            )
            
            # Input validation
            is_valid, validation_msg = validate_input(user_input)
            predict_btn = st.button("Predict Possible Conditions", type="primary", disabled=not is_valid)
            
            if not is_valid:
                st.warning(validation_msg)
            
            if predict_btn:
                # Preprocess input
                def preprocess_input(text):
                    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text).lower())
                    text = re.sub(r'\s+', ' ', text).strip()
                    words = text.split()
                    stop_words = {
                        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                        "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
                        'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it',
                        "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                        'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
                        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                        'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
                        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
                        'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                        'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
                        'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                        'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've",
                        'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
                        "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                        "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                        "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'ive', 'im',
                        'get', 'got', 'like', 'feel', 'feeling', 'still', 'even', 'one', 'two', 'three',
                        'also', 'much', 'many', 'lot', 'really', 'maybe', 'perhaps', 'sometimes', 'usually',
                        'often', 'always', 'never', 'every', 'since', 'ago', 'would', 'could', 'may', 'might',
                        'must', 'shall', 'thats', 'theres', 'whens', 'wheres', 'whys', 'hows', 'ok', 'okay',
                        'yes', 'no', 'hi', 'hello', 'hey', 'please', 'thanks', 'thank', 'sorry', 'oh', 'ah',
                        'um', 'uh', 'er', 'well', 'actually', 'basically', 'seriously', 'literally', 'right',
                        'left', 'back', 'front', 'top', 'bottom', 'side', 'sides', 'part', 'parts', 'area',
                        'areas', 'thing', 'things', 'stuff', 'kind', 'kinds', 'sort', 'sorts', 'bit', 'bits',
                        'little', 'big', 'small', 'large', 'long', 'short', 'high', 'low', 'good', 'bad',
                        'better', 'worse', 'worst', 'best', 'new', 'old', 'young', 'first', 'last', 'next',
                        'previous', 'early', 'late', 'recent', 'recently', 'already', 'yet', 'almost', 'nearly',
                        'quite', 'rather', 'somewhat', 'too', 'very', 'extremely', 'absolutely', 'completely',
                        'totally', 'utterly', 'perfectly', 'exactly', 'precisely', 'certainly', 'definitely',
                        'probably', 'possibly', 'likely', 'unlikely', 'usually', 'normally', 'typically',
                        'generally', 'especially', 'particularly', 'mainly', 'mostly', 'chiefly', 'primarily',
                        'usually', 'commonly', 'frequently', 'regularly', 'often', 'sometimes', 'occasionally',
                        'rarely', 'seldom', 'hardly', 'never', 'always', 'constantly', 'continuously'
                    }
                    filtered_words = [
                        word for word in words 
                        if word not in stop_words 
                        and len(word) > 2
                        and not word.isnumeric()
                    ]
                    return ' '.join(filtered_words)
                
                processed_input = preprocess_input(user_input)
                X_input = tfidf.transform([processed_input])
                
                # Create tabs for each model
                tab1, tab2, tab3 = st.tabs(["Random Forest", "Support Vector Machine", "Logistic Regression"])
                
                for i, (name, tab) in enumerate(zip(["Random Forest", "Support Vector Machine", "Logistic Regression"], [tab1, tab2, tab3])):
                    with tab:
                        model_data = results[name]
                        model = model_data["model"]
                        
                        if model_data["y_proba"] is not None:
                            probabilities = model.predict_proba(X_input)[0]
                            top_3 = np.argsort(probabilities)[-2:][::-1]  # Get top 3 predictions
                            
                            st.subheader(f"Prediction Results ({name})")
                            
                            for j, idx in enumerate(top_3, 1):
                                disease = le.inverse_transform([idx])[0]
                                prob = probabilities[idx]
                                
                                # Get the most important features for this prediction
                                feature_names = tfidf.get_feature_names_out()
                                
                                with st.expander(f"🏷️ {j}. {disease}"):
                                    # Show matching keywords
                                    st.write("**Key Symptoms Matched:**")
                                    disease_samples = df[df['label'] == disease]['keyword_symptoms'].values
                                    all_disease_words = ' '.join(disease_samples).split()
                                    disease_word_counts = Counter(all_disease_words)
                                    
                                    # Get the most relevant matched words
                                    user_words = set(processed_input.split())
                                    disease_words = set(all_disease_words)
                                    matched_words = user_words & disease_words
                                    
                                    # Sort by importance in the disease description
                                    matched_words = sorted(
                                        matched_words, 
                                        key=lambda x: disease_word_counts.get(x, 0), 
                                        reverse=True
                                    )[:10]  # Show top 10 most relevant matches
                                    
                                    if matched_words:
                                        st.write(", ".join(matched_words))
                                    else:
                                        st.write("General pattern match (no specific keywords identified)")
                                    
                                    # Show example symptoms
                                    st.write("**Typical Symptoms for This Condition:**")
                                    example_symptoms = disease_samples[0]
                                    if len(disease_samples) > 1:
                                        example_symptoms += "; " + disease_samples[1]
                                    st.write(example_symptoms[:300] + ("..." if len(example_symptoms) > 300 else ""))
                                    
                                    # Show additional disease info
                                    show_disease_info(disease)
                                    
                                
            
            # Data exploration
            st.sidebar.header("Data Insights")
            if st.sidebar.checkbox("Show disease distribution"):
                st.subheader("Diseases in Dataset")
                fig = px.bar(df['label'].value_counts(), 
                            title="Number of Cases per Disease",
                            labels={'value': 'Number of Cases', 'index': 'Disease'})
                st.plotly_chart(fig, use_container_width=True)
                
            if st.sidebar.checkbox("Show symptom keywords cloud"):
                st.subheader("Most Common Symptom Keywords")
                all_keywords = ' '.join(df['keyword_symptoms']).split()
                word_counts = Counter(all_keywords)
                
                # Create word cloud
                wordcloud_df = pd.DataFrame(word_counts.most_common(50), columns=['word', 'count'])
                
                fig = px.bar(wordcloud_df.head(20), 
                            x='count', y='word',
                            title="Top 20 Symptom Keywords",
                            orientation='h')
                st.plotly_chart(fig, use_container_width=True)
                
            if st.sidebar.checkbox("Show model performance details"):
                st.subheader("Model Performance Details")
                
                for name, result in results.items():
                    with st.expander(f"{name} Performance"):
                        st.write(f"Accuracy: {result['accuracy']:.2%}")
                        
                        # Classification Report
                        st.write("Classification Report:")
                        report_df = pd.DataFrame(result['report']).transpose()
                        st.dataframe(report_df.drop(columns=['support']).style.format("{:.2f}"))
                        
                        # Plot Confusion Matrix
                        fig = plot_confusion_matrix(result['y_test'], result['y_pred'], le.classes_)
                        st.plotly_chart(fig, use_container_width=True)

                        
            if st.sidebar.checkbox("Show raw data sample"):
                st.subheader("Sample of Training Data")
                st.dataframe(df.sample(10))
            
            if st.sidebar.checkbox("Compare models"):
                st.subheader("Accuracy")
                # Extract model names
                model_names = list(results.keys())

                # Initialize metric lists
                accuracies = []
                precisions = []
                recalls = []
                f1_scores = []

                # Populate the lists from results
                for name in model_names:
                    report = results[name]['report']
                    accuracy = results[name]['accuracy']
                    precision = report['weighted avg']['precision']
                    recall = report['weighted avg']['recall']
                    f1_score = report['weighted avg']['f1-score']
                    
                    accuracies.append(accuracy)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1_scores.append(f1_score)

                # Create the grouped bar chart
                fig = go.Figure(data=[
                    go.Bar(name='Accuracy', x=model_names, y=accuracies, text=[f"{v:.2%}" for v in accuracies], textposition='auto'),
                    go.Bar(name='Precision', x=model_names, y=precisions, text=[f"{v:.2%}" for v in precisions], textposition='auto'),
                    go.Bar(name='Recall', x=model_names, y=recalls, text=[f"{v:.2%}" for v in recalls], textposition='auto'),
                    go.Bar(name='F1-Score', x=model_names, y=f1_scores, text=[f"{v:.2%}" for v in f1_scores], textposition='auto')
                ])

                # Customize layout
                fig.update_layout(
                    barmode='group',
                    title='Model Performance Comparison',
                    xaxis_title='Model',
                    yaxis_title='Score',
                    yaxis=dict(range=[0, 1]),
                    plot_bgcolor='rgba(0,0,0,0)',
                    bargap=0.2
                )

                # Show plot in Streamlit
                st.plotly_chart(fig)

                f1_data = {
                    model_name: result["report"]["weighted avg"]["f1-score"]
                    for model_name, result in results.items()
                }

                st.subheader("📊 Model F1 Score Comparison (Weighted Avg)")
                st.bar_chart(pd.DataFrame.from_dict(f1_data, orient='index', columns=["F1 Score"]))
                precision_data = {
                model_name: result["report"]["weighted avg"]["precision"]
                for model_name, result in results.items()
                }

                st.subheader("🎯 Precision Comparison")
                st.bar_chart(pd.DataFrame.from_dict(precision_data, orient='index', columns=["Precision"]))

            
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.session_state.models_trained = False

if __name__ == "__main__":
    main()