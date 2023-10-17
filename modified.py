import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plot

image = st.image("sleep.png", use_column_width=True)
c1, c2 = st.columns(2)
c1.markdown("# Hello, I am Prakash KC")
c2.markdown("# This is my Webapp!")
st.write(" Explore the relationship between sleep and health through interactive data visualizations with this web app. It includes diverse plots for surface insights into how sleep duration, efficiency, timing, and quality correlate to health markers. Interactive controls allow you to highlight points of interest. The app enables discovery of connections between sleep and wellbeing. Use these insights to learn about the importance of quality sleep and improve your own habits for better health.")
# Load your dataset
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

categorical_columns = ["Gender", "Occupation", "Sleep Disorder", "BMI Category", "Blood Pressure"]
numerical_columns = [col for col in df.columns if col not in categorical_columns]

# Display the title
st.write("## Dataset for Sleep_health_and_lifestyle")

# Display the DataFrame
st.dataframe(df)

plot_type = st.sidebar.selectbox(
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
    
    x = st.sidebar.selectbox('x_axis:', numerical_columns)
    y = st.sidebar.selectbox('y_axis', numerical_columns)
    
    
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, ax=ax)
    
    st.write("### What is Linear Regression Plot?")
    st.write(" The linear regression plot shows the relationship between two variables with a straight line. It has sliders to adjust the slope and intercept. As you change the slope and intercept, the line and the mean absolute and mean squared errors update. This allows you to find the linear model that best fits the data. The interactive controls make it easy to visualize and analyze linear relationships.")
    slope = st.slider('Slope', min_value=-5.0, max_value=5.0, value=0.0)
    intercept = st.slider('Intercept', min_value=-5.0, max_value=5.0, value=0.0)
    x_values = np.array(df[x])
    y_actual = np.array(df[y])
    y_values = slope * x_values + intercept
    mae = mean_absolute_error(y_actual, y_values)
    mse = mean_squared_error(y_actual, y_values)
    sns.lineplot(x=x_values, y=y_values, color='red', ax=ax)
    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Mean Squared Error (MSE): {mse}")

elif plot_type == 'Scatter':
    st.write("### What is Scatter Plot?")
    st.write(" A scatter plot visualizes the relationship between two continuous variables. It uses dots to represent each data point, with the position showing the values for the x and y axes. The pattern of the dots reveals correlations and trends. Scatter plots are useful for identifying positive, negative, or lack of correlation between variables. Scatter plots make it easy to spot relationships in data at a glance.")
    x = st.sidebar.selectbox('x_axis:', numerical_columns)
    y = st.sidebar.selectbox('y_axis', df.columns)
    
    sns.scatterplot(data=df, x=x, y=y)
    
elif plot_type == 'Relational Plot':
    st.write("### What is Relational Plot?")
    st.write("A relational plot uses scatter points to visualize the relationship between two quantitative variables. The closer the points are to a straight line, the higher the correlation between the variables. Relational plots make it easy to spot positive, negative, or lack of correlation at a glance. Relational plots are useful for identifying trends and associations between continuous variables in a dataset.")
    x = st.sidebar.selectbox('x_axis:', df.columns)
    y = st.sidebar.selectbox('y_axis', df.columns)
    
    sns.scatterplot(data=df, x=x, y=y)

elif plot_type == 'Categorical Plot':
    st.write("### What is Categorical Plot?")
    st.write(" A categorical plot visualizes the distribution of a continuous variable across distinct categories. It divides the data into groups based on a categorical variable, and plots each group using bars or boxes to represent summary statistics. Categorical plots allow easy comparison of metrics between categories. Categorical plots are useful for understanding how a quantitative variable changes based on a categorical variable.")
    x = st.sidebar.selectbox('Select data you want to plot', df.columns)
    
    sns.histplot(data=df, x=x)    

elif plot_type == 'Line Plot':
    st.write("### What is Line Plot?")
    st.write(" A line plot displays how a continuous variable changes over time or across some sequential value. It connects a series of data points with line segments to show trends and patterns. Line plots make it easy to see increases, decreases, peaks, valleys, and anomalies in data. Line plots are useful for visualizing trends over time, demonstrating change, and comparing several lines to analyze relationships between variables.")
    x = st.sidebar.selectbox('x_axis:', df.columns)
    y = st.sidebar.selectbox('y_axis', df.columns)
    
    sns.lineplot(data=df, x=x, y=y)
    
elif plot_type == 'Bar Plot':
    st.write("### What is Bar Plot?")
    st.write("A bar plot uses rectangular bars to visualize comparisons across categories or time. The height of each bar represents the measured value. Bars can be plotted vertically or horizontally. Bar plots make it easy to compare values across groups and see patterns in data at a glance. Bar plots are useful for comparing categorical data, ranking values, and spotting trends and outliers in measurements across groups or time periods.")
    x = st.sidebar.selectbox('x_axis:', numerical_columns)
    y = st.sidebar.selectbox('y_axis', df.columns)
    
    sns.barplot(data=df, x=x, y=y)
    

elif plot_type == 'Heatmap':
    st.write("### What is Heatmap?")
    st.write(" A heat map uses color to represent values in a table of numbers. Cells are shaded different colors depending on the data, making patterns easy to spot. Darker shades represent higher values, lighter shades are lower values. Heat maps allow quick analysis of correlations, relationships, and clustering in large data sets. The color coding makes it easy to visualize complex relationships and notice intriguing patterns.")
    corr_matrix = df[numerical_columns].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

elif plot_type == 'Pair Plot':
    st.write("### What is Pair Plot?")
    st.write(" A pair plot displays the relationship between each pair of variables in a dataset using scatter plots. Each variable is plotted against all the others, allowing analysis of interactions between all the variables. The main diagonal shows the distribution of each variable. Pair plots are useful for quickly identifying correlations and relationships between all variables. Pair plots provide a broad overview of the relationships in multivariate data, making them an exploratory tool for general analysis.")
    sns.pairplot(df)


elif plot_type == '3D Plot':
    st.write("### What is 3D Plot?")
    st.write(" A 3D plot displays data points on three axes (x, y, and z) to show relationships between three variables. 3D plots reveal correlations, clusters, and patterns that may not be obvious in 2D plots. 3D plots are useful for visualizing complex multivariate data and identifying interactions between variables that are missed in 2D. They excel at bringing out hidden structures in high dimensional data.")
    x = st.sidebar.selectbox('x_axis:', numerical_columns)
    y = st.sidebar.selectbox('y_axis', numerical_columns)
    z = st.sidebar.selectbox('z_axis', numerical_columns)
    x_zoom = st.slider('X-axis Zoom', min_value=0.1, max_value=2.0, value=1.0)
    y_zoom = st.slider('Y-axis Zoom', min_value=0.1, max_value=2.0, value=1.0)

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x], df[y], df[z])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_xlim(ax.get_xlim()[0] * x_zoom, ax.get_xlim()[1] * x_zoom)
    ax.set_ylim(ax.get_ylim()[0] * y_zoom, ax.get_ylim()[1] * y_zoom)

    

st.pyplot(plt.gcf())

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Exploring a story of Sleep and Health Data")

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
    sns.boxplot(data=df, x="Stress Level", y="Sleep Duration")
    st.pyplot()
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between Stress Level and Sleep Duration. "
               "It shows that higher stress levels are associated with longer sleep durations, suggesting that stress may affect sleep duration positively.")

    st.subheader("BMI Category vs. Sleep Duration")
    sns.boxplot(data=df, x="BMI Category", y="Sleep Duration")
    st.pyplot()
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between BMI Category and Sleep Duration. "
               "It shows that individuals with normal weight tend to have better sleep duration compared to those who are overweight or obese.")

    st.subheader("Sleep Disorder vs. Sleep Duration")
    sns.boxplot(data=df, x="Sleep Disorder", y="Sleep Duration")
    st.pyplot()
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between Sleep Disorder and Sleep Duration. "
               "It shows that individuals with insomnia tend to have shorter sleep durations.")
