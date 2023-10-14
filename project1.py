import streamlit as st
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_option('deprecation.showPyplotGlobalUse', False)  # Suppress warning

c1, c2 = st.beta_columns(2)
c1.markdown("# Hello, I am Prakash KC")
c2.markdown("# This is my Webapp!")

# Load your dataset (specify the file extension, e.g., ".csv")
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# Display the title
st.write("## Dataset for Sleep_health_and_lifestyle")

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
        "Heatmap",
        "Linear Regression"
    )
)

if plot_type == 'Linear Regression':
    x = st.selectbox('x_axis:', df.columns)
    y = st.selectbox('y_axis', df.columns)
    
    slope = st.slider('Slope', min_value=-10.0, max_value=10.0, value=0.0)
    intercept = st.slider('Intercept', min_value=-10.0, max_value=10.0, value=0.0)
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    
    x_values = np.array(df[x])
    y_values = slope * x_values + intercept
    
    sns.lineplot(x=x_values, y=y_values, color='red', ax=ax)
    
    y_actual = np.array(df[y])
    mae = mean_absolute_error(y_actual, y_values)
    mse = mean_squared_error(y_actual, y_values)
    
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")
    
elif plot_type == 'Scatter':
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
st.pyplot(plt.gcf())
