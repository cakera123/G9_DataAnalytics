import streamlit as st
import pandas as pd
from model_utils import load_model, predict_risk, generate_shap_waterfall
from visualizations import create_gauge_chart, create_feature_importance_chart, create_pie_chart, create_age_risk_chart, create_confidence_histogram, create_bmi_bp_scatter

# ------------------------------------------------
# 1. Page Configuration & Layout
# ------------------------------------------------
st.set_page_config(page_title="CVD Risk Assessment", page_icon="🫀", layout="wide")

st.title("🫀 Cardiovascular Disease Risk Assessment Tool")
st.markdown("""
**Disclaimer:** This is a prototype decision-support tool for educational purposes only. 
It is not validated for clinical use. The predictions should be interpreted as a risk estimation aid, not a medical diagnosis.
""")

# ------------------------------------------------
# 2. Load the Model
# ------------------------------------------------
try:
    rf_model = load_model()
    model_loaded = True
except Exception as e:
    st.error("Error loading model. Please ensure 'random_forest_model.pkl' is in the same directory.")
    model_loaded = False

# Risk level mapping dictionary
risk_mapping = {0: "Low", 1: "Intermediary", 2: "High"}
feature_columns = ['Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI', 
                   'Abdominal Circumference (cm)', 'Total Cholesterol (mg/dL)', 
                   'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)', 'Smoking Status', 
                   'Diabetes Status', 'Physical Activity Level', 'Family History of CVD', 
                   'Systolic BP', 'Diastolic BP']

# ------------------------------------------------
# 3. Create Tabs
# ------------------------------------------------
tab1, tab2 = st.tabs(["📝 Manual Entry", "📁 Batch Upload (CSV)"])

# ================================================
# TAB 1: MANUAL ENTRY
# ================================================
with tab1:
    st.header("Patient Data Input")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=40)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=250.0, value=70.0)
        height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.70)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0)
        
    with col2:
        abd_circumference = st.number_input("Abdominal Circumference (cm)", value=90.0)
        tot_cholesterol = st.number_input("Total Cholesterol (mg/dL)", value=200.0)
        hdl = st.number_input("HDL (mg/dL)", value=50.0)
        fasting_bs = st.number_input("Fasting Blood Sugar (mg/dL)", value=100.0)
        sys_bp = st.number_input("Systolic BP", value=120.0)
        
    with col3:
        dia_bp = st.number_input("Diastolic BP", value=80.0)
        smoking = st.selectbox("Smoking Status", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        diabetes = st.selectbox("Diabetes Status", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        physical_activity = st.selectbox("Physical Activity Level", options=[0, 1, 2], format_func=lambda x: ["Low", "Moderate", "High"][x])
        family_history = st.selectbox("Family History of CVD", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    # Predict Button
    st.markdown("---")
    predict_clicked = st.button("PREDICT RISK", type="primary", use_container_width=True)
    
    if predict_clicked and model_loaded:
        # 1. Prepare Input Data
        input_data = pd.DataFrame({
            'Sex': [sex], 'Age': [age], 'Weight (kg)': [weight], 'Height (m)': [height],
            'BMI': [bmi], 'Abdominal Circumference (cm)': [abd_circumference],
            'Total Cholesterol (mg/dL)': [tot_cholesterol], 'HDL (mg/dL)': [hdl],
            'Fasting Blood Sugar (mg/dL)': [fasting_bs], 'Smoking Status': [smoking],
            'Diabetes Status': [diabetes], 'Physical Activity Level': [physical_activity],
            'Family History of CVD': [family_history], 'Systolic BP': [sys_bp], 'Diastolic BP': [dia_bp]
        })
        
        # 2. Make Prediction using model_utils
        prediction, probabilities = predict_risk(rf_model, input_data)
        
        risk_label = risk_mapping.get(prediction, "Unknown")
        pred_prob = probabilities[prediction] * 100
        
        # 3. Display Results
        st.subheader("Results & Analysis")
        
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            if prediction == 0:
                color = "green"
            elif prediction == 1:
                color = "orange"
            else:
                color = "red"
                
            st.markdown(f"""
            <div style="background-color:{color};padding:20px;border-radius:10px;text-align:center;">
                <h2 style="color:white;margin:0;">Predicted Risk Level</h2>
                <h1 style="color:white;margin:0;">{risk_label}</h1>
            </div>
            """, unsafe_allow_html=True)
            
        with res_col2:
            # Show Gauge Chart
            fig_gauge = create_gauge_chart(pred_prob, risk_label, color)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Model Insights")
        
        insight_col1, insight_col2 = st.columns([1, 1])
        
        with insight_col1:
            # Show Global Feature Importance
            st.markdown("#### What does the model care about most generally?")
            fig_importance = create_feature_importance_chart(rf_model, feature_columns)
            st.plotly_chart(fig_importance, use_container_width=True)
            
        with insight_col2:
            # Show SHAP Waterfall Plot
            st.markdown("#### Why did the model make this specific prediction?")
            st.markdown("*(Red pushes risk higher, Blue pushes risk lower)*")
            try:
                fig_shap = generate_shap_waterfall(rf_model, input_data, prediction)
                st.pyplot(fig_shap)
            except Exception as e:
                st.info("SHAP Analysis is currently unavailable for this specific configuration.")

# ================================================
# TAB 2: BATCH UPLOAD (CSV)
# ================================================
with tab2:
    st.header("Batch Prediction via CSV")
    st.markdown("Upload a CSV file containing patient records to get bulk predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None and model_loaded:
        df_batch = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(df_batch)} records.")
                         
        if all(col in df_batch.columns for col in feature_columns):
            # Predict
            features = df_batch[feature_columns]
            
            predictions = rf_model.predict(features)
            probabilities = rf_model.predict_proba(features)
            
            df_batch['Predicted_Risk_Level'] = [risk_mapping.get(p) for p in predictions]
            df_batch['Confidence_Score (%)'] = [round(prob[pred] * 100, 2) for prob, pred in zip(probabilities, predictions)]
            
            # Interactive Dataframe with Conditional Formatting
            st.subheader("Predictions Dataframe")
            def highlight_risk(val):
                color = 'salmon' if val == 'High' else 'lightgreen' if val == 'Low' else 'gold'
                return f'background-color: {color}'
            
            st.dataframe(df_batch.style.map(highlight_risk, subset=['Predicted_Risk_Level']))
            
            # Distribution Chart
            st.markdown("---")
            st.subheader("Population Insights")
            
            pop_col1, pop_col2 = st.columns(2)
            
            with pop_col1:
                fig_pie = create_pie_chart(df_batch, risk_mapping)
                st.plotly_chart(fig_pie, use_container_width=True)
                
                fig_hist = create_confidence_histogram(df_batch)
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with pop_col2:
                fig_age = create_age_risk_chart(df_batch)
                st.plotly_chart(fig_age, use_container_width=True)
                
                fig_scatter = create_bmi_bp_scatter(df_batch)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            st.markdown("---")
            # Download Button
            csv_export = df_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=csv_export,
                file_name='batch_predictions.csv',
                mime='text/csv',
                type="primary"
            )
        else:
            st.error("Uploaded CSV is missing some required columns. Please check your data format.")
