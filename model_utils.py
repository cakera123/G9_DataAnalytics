import joblib
import shap
import matplotlib.pyplot as plt
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.pkl')

def predict_risk(model, input_data):
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    return prediction, probabilities

def generate_shap_waterfall(model, input_data, prediction):
    # Explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)
    
    # Random Forest returns SHAP values for all classes. We want the one for the predicted class.
    # Depending on SHAP version, it might be shape (samples, features, classes)
    fig, ax = plt.subplots(figsize=(8, 4))
    
    try:
        if len(shap_values.shape) == 3:
            shap.plots.waterfall(shap_values[0, :, prediction], show=False)
        else:
            shap.plots.waterfall(shap_values[0], show=False)
    except Exception as e:
        # Fallback to bar plot if waterfall is not supported in the user's specific SHAP version
        try:
            if len(shap_values.shape) == 3:
                shap.plots.bar(shap_values[0, :, prediction], show=False)
            else:
                shap.plots.bar(shap_values[0], show=False)
        except:
            pass # Just return empty figure if SHAP fails entirely
            
    plt.tight_layout()
    return fig
