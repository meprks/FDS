import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('wi_cancer.csv') 



plot_type = st.selectbox(
    "Select a Plot", 
    (
        "Scatter", 
        "Relational Plot",
        "Categorical Plot"
    )
)

if plot_type == 'Scatter':
    x = st.selectbox('x_axis:', df.columns)
    y = st.selectbox('y_axis', df.columns)
    
    sns.scatterplot(data=df, x=x, y=y)
    
elif plot_type == 'Relational Plot':
    x = st.selectbox('x_axis:', df.columns)
    y = st.selectbox('y_axis', df.columns)
    
    sns.scatterplot(data=df, x=x, y=y)

elif plot_type == 'Categorical Plot':
    x = st.selectbox('Select data you want to plot', df.columns)
    
    sns.histplot(data=df, x=x)    

st.pyplot(plt.gcf())