import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Exploring Sleep and Health Data")

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

show_story = st.radio("Do you want to see the story?", ("Yes", "No"))

if show_story == "Yes":
    st.subheader("Age vs. Sleep Duration (Hue: Gender)")
    scatter_plot = alt.Chart(df).mark_circle().encode(
        x='Age',
        y='Sleep Duration',
        color=alt.Color('Gender', scale=alt.Scale(scheme='dark2')),
        tooltip=['Age', 'Sleep Duration', 'Gender']
    )
    st.altair_chart(scatter_plot, use_container_width=True)

    st.subheader("Physical Activity Level vs. Sleep Duration")
    scatter_plot = px.scatter(df, x="Physical Activity Level", y="Sleep Duration", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(scatter_plot)

    st.subheader("Stress Level vs. Sleep Duration")
    box_plot = px.box(df, x="Stress Level", y="Sleep Duration", color="Stress Level", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(box_plot)

    st.subheader("BMI Category vs. Sleep Duration")
    box_plot = px.box(df, x="BMI Category", y="Sleep Duration", color="BMI Category", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(box_plot)

    st.subheader("Sleep Disorder vs. Sleep Duration")
    box_plot = px.box(df, x="Sleep Disorder", y="Sleep Duration", color="Sleep Disorder", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(box_plot)
