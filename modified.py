import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plot

c1, c2 = st.columns(2)
c1.markdown("# Hello, I am Prakash KC")
c2.markdown("# This is my Webapp!")

# Load your dataset
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

categorical_columns = ["Gender", "Occupation", "Sleep Disorder", "BMI Category", "Blood Pressure"]
numerical_columns = [col for col in df.columns if col not in categorical_columns]

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
        "Linear Regression",
        "Pair Plot",
        "3D Plot",
    )
)


if plot_type == 'Linear Regression':
    st.write("### Linear Regression Plot")
    st.sidebar.write("ioi")
    x = st.selectbox('x_axis:', numerical_columns)
    y = st.selectbox('y_axis', numerical_columns)
    
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
    st.write("### Scatter Plot")
    x = st.selectbox('x_axis:', df.columns)
    y = st.selectbox('y_axis', df.columns)
    
    sns.scatterplot(data=df, x=x, y=y)
    
elif plot_type == 'Relational Plot':
    st.write("### Relational Plot")
    x = st.selectbox('x_axis:', df.columns)
    y = st.selectbox('y_axis', df.columns)
    
    sns.scatterplot(data=df, x=x, y=y)

elif plot_type == 'Categorical Plot':
    st.write("### Categorical Plot")
    x = st.selectbox('Select data you want to plot', df.columns)
    
    sns.histplot(data=df, x=x)    

elif plot_type == 'Line Plot':
    st.write("### Line Plot")
    x = st.selectbox('x_axis:', df.columns)
    y = st.selectbox('y_axis', df.columns)
    
    sns.lineplot(data=df, x=x, y=y)
    
elif plot_type == 'Bar Plot':
    st.write("### Bar Plot")
    x = st.selectbox('x_axis:', df.columns)
    y = st.selectbox('y_axis', df.columns)
    
    sns.barplot(data=df, x=x, y=y)
    

elif plot_type == 'Heatmap':
    st.write("### Heatmap")
    corr_matrix = df[numerical_columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

elif plot_type == 'Pair Plot':
    st.write("### Pair Plot")
    sns.pairplot(df)
    st.pyplot(plt.gcf())

elif plot_type == '3D Plot':
    st.write("### 3D Plot")
    x = st.selectbox('x_axis:', numerical_columns)
    y = st.selectbox('y_axis', numerical_columns)
    z = st.selectbox('z_axis', numerical_columns)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x], df[y], df[z])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)

    st.pyplot(fig)

st.pyplot(plt.gcf())
