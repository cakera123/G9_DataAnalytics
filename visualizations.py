import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_gauge_chart(pred_prob, risk_label, color):
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_prob,
        title={'text': "Model Confidence Score (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 100], 'color': "lightgray"}
            ]
        }
    ))
    fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig_gauge

def create_radar_chart(input_data):
    # Patient vs Standard Healthy Baseline
    categories = ['BMI', 'Systolic BP', 'Diastolic BP', 'Total Chol', 'Fasting BS']
    baseline = [22, 120, 80, 200, 100] # Standard healthy clinical limits
    
    patient_values = [
        input_data['BMI'].iloc[0],
        input_data['Systolic BP'].iloc[0],
        input_data['Diastolic BP'].iloc[0],
        input_data['Total Cholesterol (mg/dL)'].iloc[0],
        input_data['Fasting Blood Sugar (mg/dL)'].iloc[0]
    ]
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=baseline,
        theta=categories,
        fill='toself',
        name='Healthy Baseline',
        line_color='green'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=patient_values,
        theta=categories,
        fill='toself',
        name='Patient Data',
        line_color='red'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(patient_values), max(baseline)) * 1.2]
            )),
        showlegend=True,
        title="Patient vs Healthy Clinical Limits",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def create_feature_importance_chart(model, feature_names):
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df_imp = df_imp.sort_values(by='Importance', ascending=True).tail(5) # Top 5 factors
    
    fig = px.bar(df_imp, x='Importance', y='Feature', orientation='h', 
                 title="Global Feature Importance (Top 5 Factors)",
                 color='Importance', color_continuous_scale='Blues')
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig

def create_pie_chart(df_batch, risk_mapping):
    dist_counts = df_batch['Predicted_Risk_Level'].value_counts().reset_index()
    dist_counts.columns = ['Risk Level', 'Count']
    
    fig_pie = px.pie(dist_counts, names='Risk Level', values='Count', 
                     title="Risk Level Distribution",
                     color='Risk Level',
                     color_discrete_map={'High':'salmon', 'Intermediary':'gold', 'Low':'lightgreen'})
    return fig_pie

def create_age_risk_chart(df_batch):
    df_chart = df_batch.copy()
    bins = [0, 30, 40, 50, 60, 120]
    labels = ['<30', '30-40', '40-50', '50-60', '60+']
    df_chart['Age Group'] = pd.cut(df_chart['Age'], bins=bins, labels=labels, right=False)
    
    age_risk = df_chart.groupby(['Age Group', 'Predicted_Risk_Level'], observed=False).size().reset_index(name='Count')
    
    fig = px.bar(age_risk, x='Age Group', y='Count', color='Predicted_Risk_Level',
                 title="Risk Distribution by Age Group",
                 color_discrete_map={'High':'salmon', 'Intermediary':'gold', 'Low':'lightgreen'},
                 barmode='stack')
    return fig

def create_confidence_histogram(df_batch):
    fig = px.histogram(df_batch, x='Confidence_Score (%)', nbins=10,
                       title="Confidence Score Distribution",
                       color_discrete_sequence=['#636EFA'])
    return fig

def create_bmi_bp_scatter(df_batch):
    fig = px.scatter(df_batch, x='BMI', y='Systolic BP', color='Predicted_Risk_Level',
                     title="Patient Risk Clusters (BMI vs Systolic BP)",
                     color_discrete_map={'High':'salmon', 'Intermediary':'gold', 'Low':'lightgreen'},
                     hover_data=['Age', 'Confidence_Score (%)'])
    return fig
