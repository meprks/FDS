import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

st.markdown("# Hello, I am Prakash KC and This is my Webapp!")

# Load your dataset
df = pd.read_csv("metamaterials.csv")

# Display the title
st.write("## Dataset for MetaMaterials")

# Display the DataFrame
st.dataframe(df)

plot_type = st.selectbox(
    "Select a Plot", 
    (
        "Scatter", 
        "Relational Plot",
        "Categorical Plot",
        "Line Plot",
        "Bar Plot",
        "Heatmap"
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

elif plot_type == 'Line Plot':
    x = st.selectbox('x_axis:', df.columns)
    y = st.selectbox('y_axis', df.columns)
    
    sns.lineplot(data=df, x=x, y=y)
    
elif plot_type == 'Bar Plot':
    x = st.selectbox('x_axis:', df.columns)
    y = st.selectbox('y_axis', df.columns)
    
    sns.barplot(data=df, x=x, y=y)
    

elif plot_type == 'Heatmap':
    # Assuming you want a heatmap of correlations between numerical columns
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    
st.pyplot(plt.gcf())
