import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Exploring Sleep and Health Data")

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

show_story = st.radio("Do you want to see the story?", ("Yes", "No"))

if show_story == "Yes":
    st.subheader("Age vs. Sleep Duration (Hue: Gender)")
    joint_plot = sns.jointplot(data=df, x="Age", y="Sleep Duration", hue="Gender", kind="scatter")
    st.pyplot(joint_plot.fig)
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between Age and Sleep Duration, "
               "color-coded by Gender. It shows that females with a high age group have a higher sleep duration than males. "
               "The average sleep duration for males generally lies between 7 and 8 hours, while for females, it is between 8 and 9 hours.")

    st.subheader("Physical Activity Level vs. Sleep Duration")
    sns.regplot(data=df, x="Physical Activity Level", y="Sleep Duration")
    st.pyplot()
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between Physical Activity Level and Sleep Duration. "
               "It shows a positive correlation, indicating that people with high physical activity levels tend to have longer sleep durations.")
    
    st.subheader("Stress Level vs. Sleep Duration")
    colors = ["red", "green", "blue"]
    sns.boxplot(data=df, x="Stress Level", y="Sleep Duration", color=colors)
    st.pyplot()
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between Stress Level and Sleep Duration. "
               "It shows that higher stress levels are associated with longer sleep durations, suggesting that stress may affect sleep duration positively.")

    st.subheader("BMI Category vs. Sleep Duration")
    colors = ["red", "green", "blue"]
    sns.boxplot(data=df, x="BMI Category", y="Sleep Duration",color=colors)
    st.pyplot()
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between BMI Category and Sleep Duration. "
               "It shows that individuals with normal weight tend to have better sleep duration compared to those who are overweight or obese.")

    st.subheader("Sleep Disorder vs. Sleep Duration")
    colors = ["red", "green", "blue"]
    sns.boxplot(data=df, x="Sleep Disorder", y="Sleep Duration",color=colors)
    st.pyplot()
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between Sleep Disorder and Sleep Duration. "
               "It shows that individuals with insomnia tend to have shorter sleep durations.")
