import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Exploring Sleep and Health Data")

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv") 

with st.expander("See plots"):

  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Age vs Sleep Duration")
    fig = sns.jointplot(data=df, x="Age", y="Sleep Duration", hue="Gender", kind="scatter")
    st.pyplot(fig)

    st.subheader("Physical Activity vs Sleep Duration")
    fig2 = sns.regplot(data=df, x="Physical Activity Level", y="Sleep Duration")
    st.pyplot(fig2)

  with col2:
    st.subheader("Stress Level vs Sleep Duration")
    fig3 = sns.boxplot(data=df, x="Stress Level", y="Sleep Duration") 
    st.pyplot(fig3)

    st.subheader("BMI Category vs Sleep Duration")
    fig4 = sns.boxplot(data=df, x="BMI Category", y="Sleep Duration")
    st.pyplot(fig4)
    
    st.subheader("Sleep Disorder vs Sleep Duration")
    fig5 = sns.boxplot(data=df, x="Sleep Disorder", y="Sleep Duration")
    st.pyplot(fig5)
