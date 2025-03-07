import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class VisualiserTool:
    def __init__(self, df):
        self.df = df

    def plot_gender_distribution(self):
        gender_counts = self.df['Gender'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=gender_counts.index, values=gender_counts.values, hole=.3)])
        fig.update_layout(title_text='Gender Distribution', transition_duration=500)
        fig.show()

    def plot_medical_condition_distribution(self):
        condition_counts = self.df['Medical Condition'].value_counts()
        fig = px.bar(x=condition_counts.index, y=condition_counts.values, labels={'x': 'Medical Condition', 'y': 'Number of Patients'})
        fig.update_layout(title_text='Number of Patients per Medical Condition', transition_duration=500)
        fig.show()

    def plot_treatment_severity_distribution(self):
        treatment_counts = self.df['Treatments'].value_counts()
        fig = go.Figure(data=[go.Pie(labels=treatment_counts.index, values=treatment_counts.values, hole=.3)])
        fig.update_layout(title_text='Treatment Severity Distribution', transition_duration=500)
        fig.show()

    def plot_treatment_duration(self):
        self.df['Admit Date'] = pd.to_datetime(self.df['Admit Date'])
        self.df['Discharge Date'] = pd.to_datetime(self.df['Discharge Date'])
        self.df['Treatment Duration'] = (self.df['Discharge Date'] - self.df['Admit Date']).dt.days

        # Filter the dataset to show records only for the year 2024
        recent_df = self.df[self.df['Admit Date'].dt.year == 2024]

        fig = px.line(recent_df, x='Admit Date', y='Treatment Duration', labels={'Admit Date': 'Admit Date', 'Treatment Duration': 'Treatment Duration (days)'})
        fig.update_layout(title_text='Treatment Duration Over Time', transition_duration=500)
        fig.show()