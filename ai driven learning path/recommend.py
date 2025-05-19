import pandas as pd
import joblib

def recommend_learning_path(student_input):
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('kmeans_model.pkl')
    clustered_df = pd.read_csv('students_clustered.csv')

    input_scaled = scaler.transform([student_input])
    cluster = model.predict(input_scaled)[0]

    group = clustered_df[clustered_df['cluster'] == cluster]
    recommended_course = group['Course_Name'].mode()[0]

    return recommended_course
