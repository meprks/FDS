import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import zscore
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plot
import altair as alt
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)
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
        "3D Plot",
        "Violin Plot",
        "Area Chart",
        "Facet Grid",


    )
)

if plot_type == 'Scatter':
    st.write("### What is Scatter Plot?")
    st.write(" A scatter plot visualizes the relationship between two continuous variables. It uses dots to represent each data point, with the position showing the values for the x and y axes. The pattern of the dots reveals correlations and trends. Scatter plots are useful for identifying positive, negative, or lack of correlation between variables. Scatter plots make it easy to spot relationships in data at a glance.")
    x = st.sidebar.selectbox('x_axis:', numerical_columns)
    y = st.sidebar.selectbox('y_axis', numerical_columns)
    origin = st.sidebar.selectbox('Origin:', categorical_columns)  # Add this line
    scatter_plot = alt.Chart(df).mark_circle().encode(
        x=x,
        y=y,
        color=alt.Color(origin + ':N', scale=alt.Scale(scheme='set1')),  # Modify this line
        tooltip=[x, y]
    )
    st.altair_chart(scatter_plot, use_container_width=True)


elif plot_type == 'Relational Plot':
    st.write("### What is Relational Plot?")
    st.write("A relational plot uses scatter points to visualize the relationship between two quantitative variables. The closer the points are to a straight line, the higher the correlation between the variables. Relational plots make it easy to spot positive, negative, or lack of correlation at a glance. Relational plots are useful for identifying trends and associations between continuous variables in a dataset.")
    x = st.sidebar.selectbox('x_axis:', numerical_columns)
    y = st.sidebar.selectbox('y_axis', numerical_columns)
    scatter_plot = px.scatter(df, x=x, y=y, color_continuous_scale=px.colors.sequential.Plasma)
    st.plotly_chart(scatter_plot)

elif plot_type == 'Categorical Plot':
    st.write("### What is Categorical Plot?")
    st.write(" A categorical plot visualizes the distribution of a continuous variable across distinct categories. It divides the data into groups based on a categorical variable, and plots each group using bars or boxes to represent summary statistics. Categorical plots allow easy comparison of metrics between categories. Categorical plots are useful for understanding how a quantitative variable changes based on a categorical variable.")
    
    x = st.sidebar.selectbox('Select data you want to plot', categorical_columns)
    origin = st.sidebar.selectbox('Origin:', categorical_columns)  # Add this line
    bar_plot = alt.Chart(df).mark_bar().encode(
        x=alt.X(x),
        y='count()',
        color=alt.Color(origin + ':N', scale=alt.Scale(scheme='set2'))  # Modify this line
    )
    st.altair_chart(bar_plot, use_container_width=True)


elif plot_type == 'Line Plot':
    st.write("### What is Line Plot?")
    st.write(" A line plot displays how a continuous variable changes over time or across some sequential value. It connects a series of data points with line segments to show trends and patterns. Line plots make it easy to see increases, decreases, peaks, valleys, and anomalies in data. Line plots are useful for visualizing trends over time, demonstrating change, and comparing several lines to analyze relationships between variables.")
    
    x = st.sidebar.selectbox('x_axis:', numerical_columns)
    y = st.sidebar.selectbox('y_axis', numerical_columns)
    line_plot = px.line(df, x=x, y=y, color_discrete_sequence=px.colors.sequential.Plasma_r)
    st.plotly_chart(line_plot)

elif plot_type == 'Bar Plot':
    st.write("### What is Bar Plot?")
    st.write("A bar plot uses rectangular bars to visualize comparisons across categories or time. The height of each bar represents the measured value. Bars can be plotted vertically or horizontally. Bar plots make it easy to compare values across groups and see patterns in data at a glance. Bar plots are useful for comparing categorical data, ranking values, and spotting trends and outliers in measurements across groups or time periods.")
    
    x = st.sidebar.selectbox('x_axis:', numerical_columns)
    y = st.sidebar.selectbox('y_axis', numerical_columns)
    bar_plot = px.bar(df, x=x, y=y, color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(bar_plot)

elif plot_type == 'Heatmap':
    st.write("### What is Heatmap?")
    st.write(" A heat map uses color to represent values in a table of numbers. Cells are shaded different colors depending on the data, making patterns easy to spot. Darker shades represent higher values, lighter shades are lower values. Heat maps allow quick analysis of correlations, relationships, and clustering in large data sets. The color coding makes it easy to visualize complex relationships and notice intriguing patterns.")
    
    corr_matrix = df[numerical_columns].corr()
    heatmap = px.imshow(corr_matrix, color_continuous_scale='RdBu')
    st.plotly_chart(heatmap)

elif plot_type == '3D Plot':
    st.write("### What is 3D Plot?")
    st.write(" A 3D plot displays data points on three axes (x, y, and z) to show relationships between three variables. 3D plots reveal correlations, clusters, and patterns that may not be obvious in 2D plots. 3D plots are useful for visualizing complex multivariate data and identifying interactions between variables that are missed in 2D. They excel at bringing out hidden structures in high dimensional data.")
    
    x = st.sidebar.selectbox('x_axis:', numerical_columns)
    y = st.sidebar.selectbox('y_axis', numerical_columns)
    z = st.sidebar.selectbox('z_axis', numerical_columns)
    
    fig = px.scatter_3d(df, x=x, y=y, z=z, color=z, color_continuous_scale='Bluered_r')
    
    st.plotly_chart(fig)

elif plot_type == 'Violin Plot':
    st.write("### What is Violin Plot?")
    st.write(" A violin plot is similar to a box plot, but shows more information about the shape and distribution of the data. In a violin plot, the width of the shaded area represents the density of points at that value. Wider sections correspond to a higher density of points. Unlike box plots which only show summary statistics like median and quartiles, violin plots show the full range of data.")
    
    x = st.sidebar.selectbox('x_axis:', categorical_columns)
    y = st.sidebar.selectbox('y_axis', numerical_columns)
    violin_plot = px.violin(df, x=x, y=y)
    st.plotly_chart(violin_plot)

elif plot_type == 'Area Chart':
    st.write("### What is Area Chart?")
    st.write(" An area chart is a type of graph that displays data as filled shapes or areas connected together. The filled area between the x-axis and the line represents the magnitude of the values being plotted. Area charts are commonly used to visualize how a metric or value changes over time. The filled area helps emphasize the total size and scope")
    

    x = st.sidebar.selectbox('x_axis:', numerical_columns)
    y = st.sidebar.selectbox('y_axis', numerical_columns)
    area_chart = alt.Chart(df).mark_area().encode(
        x=x,
        y=y,
    )
    st.altair_chart(area_chart, use_container_width=True)
elif plot_type == 'Facet Grid':
    st.write("### What is Facet Grid?")
    st.write(" A facet grid is a matrix layout for small multiple charts that split a dataset into subsets defined by categorical variables. Facet grids arrange multiple mini-charts in a grid where each row or column represents one variable.")
    
    x = st.sidebar.selectbox('x_axis:', numerical_columns)
    y = st.sidebar.selectbox('y_axis', numerical_columns)
    facet_column = st.sidebar.selectbox('Facet Column:', categorical_columns)
    facet_row = st.sidebar.selectbox('Facet Row:', categorical_columns)
    facet_grid = px.scatter(df, x=x, y=y, facet_col=facet_column, facet_row=facet_row)
    st.plotly_chart(facet_grid)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Exploring Sleep and Health Data")
show_story = st.radio("### Do you want to see the story?", ("Yes", "No"))

if show_story == "Yes":
    st.subheader("Age vs. Sleep Duration (Hue: Gender)")
    scatter_plot = alt.Chart(df).mark_circle().encode(
        x='Age',
        y='Sleep Duration',
        color=alt.Color('Gender', scale=alt.Scale(scheme='dark2')),
        tooltip=['Age', 'Sleep Duration', 'Gender']
    )
    st.altair_chart(scatter_plot, use_container_width=True)
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between Age and Sleep Duration, "
               "color-coded by Gender. It shows that females with a high age group have a higher sleep duration than males. "
               "The average sleep duration for males generally lies between 7 and 8 hours, while for females, it is between 8 and 9 hours.")


    st.subheader("Physical Activity Level vs. Sleep Duration")
    scatter_plot = px.scatter(df, x="Physical Activity Level", y="Sleep Duration", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(scatter_plot)
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the correlation between data. "
               "It shows a positive correlation for physical activity level and sleep duration, indicating that people with high physical activity levels tend to have longer sleep durations.")
    

    st.subheader("Stress Level vs. Sleep Duration")
    box_plot = px.box(df, x="Stress Level", y="Sleep Duration", color="Stress Level", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(box_plot)
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between Stress Level and Sleep Duration. "
               "It shows that higher stress levels are associated with longer sleep durations, suggesting that stress may affect sleep duration positively.")


    st.subheader("BMI Category vs. Sleep Duration")
    box_plot = px.box(df, x="BMI Category", y="Sleep Duration", color="BMI Category", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(box_plot)
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between BMI Category and Sleep Duration. "
               "It shows that individuals with normal weight tend to have better sleep duration compared to those who are overweight or obese.")


    st.subheader("Sleep Disorder vs. Sleep Duration")
    box_plot = px.box(df, x="Sleep Disorder", y="Sleep Duration", color="Sleep Disorder", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(box_plot)
    st.write("### What does the Plot show?")
    st.write("The plot visualizes the relationship between Sleep Disorder and Sleep Duration. "
               "It shows that individuals with insomnia tend to have shorter sleep durations.")
