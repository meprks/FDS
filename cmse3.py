import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

df=pd.read_csv('wi_cancer.csv')

#Can select the columns from the side bar 
x_lbl = st.sidebar.selectbox('x axis: ', df.columns)
y_lbl = st.sidebar.selectbox('y axis: ', df.drop(columns=[x_lbl]).columns)
hue_dia = st.sidebar.selectbox('hue: ', df.drop(columns = [x_lbl, y_lbl]).columns)


st.header("Relational, Distribution and Categorical Plot")
sd = st.selectbox(
    "Select a Plot", #Drop Down Menu Name
    (
        "Relational Plot", #First option in menu
        "Distribution Plot",
        "Categorical Plot"
    )
)

fig, ax = plt.subplots()

if sd == "Relational Plot":
    sns.relplot(data = df, x = x_lbl, y = y_lbl, hue = hue_dia, style = hue_dia, ax = ax)

elif sd == "Distribution Plot":
    sns.histplot(data = df, x = x_lbl, y = y_lbl, hue = hue_dia, ax = ax)

elif sd == "Categorical Plot":
    sns.catplot(data = df, x = x_lbl, y = y_lbl, hue = hue_dia, split = True, palette = "pastel", ax = ax)
    

st.pyplot(plt.gcf())