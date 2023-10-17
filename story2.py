import streamlit as st
import pandas as pd 
import plotly.express as px
import altair as alt

st.title('Exploring Sleep Dataset')

df = pd.read_csv('sleep_dataset.csv')

if st.checkbox('Show Plots'):

  # Age vs Sleep Duration 
  fig = px.scatter(df, x='Age', y='Sleep Duration', color='Gender', title='Age vs Sleep Duration')
  st.plotly_chart(fig)

  # Physical Activity vs Sleep Duration
  chart = alt.Chart(df).mark_circle().encode(
      x='Physical Activity',
      y='Sleep Duration')
  st.altair_chart(chart)

  # Stress Level vs Sleep Duration
  fig = px.box(df, x='Stress Level', y='Sleep Duration')
  st.plotly_chart(fig)

  # BMI vs Sleep Duration
  chart = alt.Chart(df).mark_boxplot().encode(
      x='BMI', y='Sleep Duration')
  st.altair_chart(chart)

  # Sleep Disorder vs Sleep Duration
  fig = px.violin(df, x='Sleep Disorder', y='Sleep Duration', color='Sleep Disorder', box=True)
  st.plotly_chart(fig)

  # Gender vs Sleep Duration 
  fig = px.violin(df, x='Gender', y='Sleep Duration', color='Gender', box=True)
  st.plotly_chart(fig)
