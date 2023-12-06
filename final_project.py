import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
import altair as alt
import plotly.express as px
from PIL import Image
import hiplot as hip
import random
from scipy import linalg

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay,f1_score



image = st.image("sleep.png", use_column_width=True)

st.write(" The webapp is designed to explore the relationship between sleep and health through interactive data visualizations and predict the quality of sleep using several machine learning technique. It includes diverse plots for surface insights into how sleep duration, efficiency, timing, and quality correlate to health markers. Interactive controls allow you to highlight points of interest. The app enables discovery of connections between sleep and wellbeing. Use these insights to learn about the importance of quality sleep and improve your own habits for better health.")

df = pd.read_csv("sleep.csv")
categorical_columns = ["Gender", "Occupation", "Sleep Disorder", "BMI Category"]
excluded_columns = categorical_columns + ["Person ID"]+["Blood Pressure"]
numerical_columns = [col for col in df.columns if col not in excluded_columns]
author_tab, data_tab, eda_tab, story_tab, pca_tab, model_tab, class_tab= st.tabs(["About Author","About Dataset", "Basic EDA", "Story","PCA","Model Prediction","Classification"])
with author_tab:
    
    st.image('prks.jpg', caption='Prakash KC') 
    st.write("Hi, I am Prakash K.C. I completed my BSc in Mechanical Engineering. Currently, I am pursuing my graduate study in Mechanical engineering department at Michigan State University. I am working as a research assistant at the FMATH research group. I am an enthusiastic, curious, and industrious person.")
    col1, col2, col3,col4,col5 = st.columns([2, 2 ,2, 2, 2])
    col1.markdown("[My GitHub](https://github.com/meprks)")  
    col3.markdown("[My Website](https://sites.google.com/view/meprks/home)")  
    col5.markdown("[My LinkedIn](https://www.linkedin.com/in/prks/)")


with data_tab:
    st.write("## Dataset for Sleep_health_and_lifestyle")
    st.write("The Sleep and Health Dataset is a comprehensive collection of data sourced from various studies and surveys, focused on exploring the relationship between sleep patterns and overall health based upon several category. Hosted on Kaggle, this dataset is an invaluable resource for researchers, data scientists, and healthcare professionals interested in understanding the impact of sleep on human health..")
    c1, c2, c3 = st.columns([3, 4, 4])
    c1.write("### Dataset")
    b1 = c1.button(":green[Show dataset]")
    if b1:
        st.dataframe(df)
    c2.write("### Numerical Columns")    
    b2 = c2.button(":green[Show Numerical columns]")
    if b2:
        st.text(', '.join(numerical_columns))
        st.dataframe(df[numerical_columns])
    c3.write("### Categorical Columns")
    b3 = c3.button(":green[Show Categorical columns]")
    if b3:
         st.text(', '.join(categorical_columns))
         st.dataframe(df[categorical_columns])
with eda_tab:
    st.write("## In this section, we will explore several visualization of dataset")
    plot_type = st.selectbox(
    "Select a Plot",
    (
        "Scatter",
        "Relational Plot",
        "Categorical Plot",
        "Bar Plot",
        "Heatmap",
        "3D Plot",
        "Violin Plot",
        "Area Chart",
        "Facet Grid",


    )
)

    if plot_type == 'Scatter':
        x = st.selectbox('x_axis:', numerical_columns)
        y = st.selectbox('y_axis', numerical_columns)
        origin = st.selectbox('Origin:', categorical_columns)  
        scatter_plot = alt.Chart(df).mark_circle().encode(x=x,y=y,
        color=alt.Color(origin + ':N', scale=alt.Scale(scheme='set1')),  
        tooltip=[x, y])
        st.altair_chart(scatter_plot, use_container_width=True)


    elif plot_type == 'Relational Plot':
        x = st.selectbox('x_axis:', numerical_columns)
        y = st.selectbox('y_axis', numerical_columns)
        scatter_plot = px.scatter(df, x=x, y=y, color_continuous_scale=px.colors.sequential.Plasma)
        st.plotly_chart(scatter_plot)

    elif plot_type == 'Categorical Plot':
        x = st.selectbox('Select data you want to plot', categorical_columns)
        origin = st.selectbox('Origin:', categorical_columns) 
        bar_plot = alt.Chart(df).mark_bar().encode(
        x=alt.X(x),
        y='count()',
        color=alt.Color(origin + ':N', scale=alt.Scale(scheme='set2'))  
    )
        st.altair_chart(bar_plot, use_container_width=True)

    elif plot_type == 'Bar Plot':
       x = st.selectbox('x_axis:', numerical_columns)
       y = st.selectbox('y_axis', numerical_columns)
       bar_plot = px.bar(df, x=x, y=y, color_discrete_sequence=px.colors.sequential.Viridis)
       st.plotly_chart(bar_plot)

    elif plot_type == 'Heatmap':
        corr_matrix = df[numerical_columns].corr()
        heatmap = px.imshow(corr_matrix, color_continuous_scale='RdBu')
        st.plotly_chart(heatmap)

    elif plot_type == '3D Plot':
        x = st.selectbox('x_axis:', numerical_columns)
        y = st.selectbox('y_axis', numerical_columns)
        z = st.selectbox('z_axis', numerical_columns)
    
        fig = px.scatter_3d(df, x=x, y=y, z=z, color=z, color_continuous_scale='Bluered_r')
    
        st.plotly_chart(fig)

    elif plot_type == 'Violin Plot':
        x = st.selectbox('x_axis:', categorical_columns)
        y = st.selectbox('y_axis', numerical_columns)
        violin_plot = px.violin(df, x=x, y=y)
        st.plotly_chart(violin_plot)

    elif plot_type == 'Area Chart':
        x = st.selectbox('x_axis:', numerical_columns)
        y = st.selectbox('y_axis', numerical_columns)
        area_chart = alt.Chart(df).mark_area().encode(
        x=x,
        y=y,
    )
        st.altair_chart(area_chart, use_container_width=True)
    elif plot_type == 'Facet Grid':
        x = st.selectbox('x_axis:', numerical_columns)
        y = st.selectbox('y_axis', numerical_columns)
        facet_column = st.selectbox('Facet Column:', categorical_columns)
        facet_row = st.selectbox('Facet Row:', categorical_columns)
        facet_grid = px.scatter(df, x=x, y=y, facet_col=facet_column, facet_row=facet_row)
        st.plotly_chart(facet_grid)  
    st.write("### HiPlot")
    b1 = st.button(":green[Show HiPlot]")
    if b1:
        exp = hip.Experiment.from_dataframe(df)
        hiplot_html = exp.to_html()
        st.components.v1.html(exp.to_html(), width=1000, height=500, scrolling=True) 
    if st.button(":green[Hide HiPlot]"):
        b1 = False

with story_tab:
    st.title("Exploring Sleep and Health Data")
    st.subheader("Age vs. Sleep Duration (Hue: Gender)")
    scatter_plot = alt.Chart(df).mark_circle().encode(
        x='Age',
        y='Sleep Duration',
        color=alt.Color('Gender', scale=alt.Scale(scheme='dark2')),
        tooltip=['Age', 'Sleep Duration', 'Gender']
    )
    st.altair_chart(scatter_plot, use_container_width=True)
    st.write(" It shows that females with a high age group have a higher sleep duration than males. "
               "The average sleep duration for males generally lies between 7 and 8 hours, while for females, it is between 8 and 9 hours.")


    st.subheader("Physical Activity Level vs. Sleep Duration")
    scatter_plot = px.scatter(df, x="Physical Activity Level", y="Sleep Duration", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(scatter_plot)
    st.write("It shows a positive correlation for physical activity level and sleep duration, indicating that people with high physical activity levels tend to have longer sleep durations.")
    

    st.subheader("Stress Level vs. Sleep Duration")
    box_plot = px.box(df, x="Stress Level", y="Sleep Duration", color="Stress Level", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(box_plot)
    st.write("It shows that higher stress levels are associated with longer sleep durations, suggesting that stress may affect sleep duration positively.")


    st.subheader("BMI Category vs. Sleep Duration")
    box_plot = px.box(df, x="BMI Category", y="Sleep Duration", color="BMI Category", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(box_plot)
    st.write("It shows that individuals with normal weight tend to have better sleep duration compared to those who are overweight or obese.")


    st.subheader("Sleep Disorder vs. Sleep Duration")
    box_plot = px.box(df, x="Sleep Disorder", y="Sleep Duration", color="Sleep Disorder", color_discrete_sequence=px.colors.qualitative.Dark24)
    st.plotly_chart(box_plot)
    st.write("It shows that individuals with insomnia tend to have shorter sleep durations.")

with pca_tab:
    excluded = categorical_columns + ["Person ID"]+["Blood Pressure"]+["Sleep Duration"]
    numerical = [col for col in df.columns if col not in excluded]
    st.title("PCA Analysis on multiple data")
    x = st.selectbox("Select Category for PCA", ["Age", "Quality of Sleep"])
    y = st.selectbox("Select Category for PCA", ["Physical Activity Level", "Stress Level" ])
    z = st.selectbox("Select Category for PCA", ["Heart Rate", "Daily Steps" ])
    selected_columns = df[[x, y, z]]
    centered_data = selected_columns - selected_columns.mean()
    U, Sigma, VT = np.linalg.svd(centered_data)
    m = len(U)
    n = len(VT)
    sigma = np.zeros([m,n])
    sigma1 = np.copy(sigma)
    for i in range(1):
        sigma1[i,i] = Sigma[i]
    X_1D = U@sigma1@VT
    sigma2 = np.copy(sigma)
    for i in range(2):
        sigma2[i,i] = Sigma[i]
    X_2D = U@sigma2@VT

    fig_3D = px.scatter_3d(centered_data, x=x, y=y, z=z, 
                        title='3D Scatter Plot of Data')
    st.plotly_chart(fig_3D)
    fig_2D = px.scatter_3d(pd.DataFrame(X_2D, columns=[x, y, z]), 
                       x=x, y=y, z=z, 
                       title='2D Scatter Plot of Data')
    st.plotly_chart(fig_2D)
    fig_1D = px.scatter_3d(pd.DataFrame(X_1D, columns=[x, y, z]), 
                       x=x, y=y, z=z, 
                       title='1D Scatter Plot of Data')

    st.plotly_chart(fig_1D)

with model_tab:
    st.write("## In this section, you can predict you sleep duration and quality based upon  Age, Physical Activity Level, Stress Level, Heart Rate, Daily Steps.")
   
    X = df[['Age', 'Physical Activity Level', 'Stress Level', 'Heart Rate', 'Daily Steps']]
    y_duration = df['Sleep Duration']
    y_quality = df['Quality of Sleep']
    X_train_dur, X_test_dur, y_train_dur, y_test_dur = train_test_split(X, y_duration, test_size=0.2, random_state=42)
    X_train_qual, X_test_qual, y_train_qual, y_test_qual = train_test_split(X, y_quality, test_size=0.2, random_state=42)
    model_duration = LinearRegression()
    model_quality = LinearRegression()
    model_duration.fit(X_train_dur, y_train_dur)
    model_quality.fit(X_train_qual, y_train_qual)
    predictions_duration = model_duration.predict(X_test_dur)
    predictions_quality = model_quality.predict(X_test_qual)
    mse_duration = mean_squared_error(y_test_dur, predictions_duration)
    mse_quality = mean_squared_error(y_test_qual, predictions_quality)
    st.write(f"Mean Squared Error for Sleep Duration:", mse_duration)
    st.write(f"Mean Squared Error for Sleep Quality:", mse_quality)
    st.title("Predict Sleep Parameters")
    age = st.slider('Age', min_value=10, max_value=80, value=25)
    physical_activity = st.slider('Physical Activity Level', min_value=20, max_value=90, value=25)
    stress_level = st.slider('Stress Level', min_value=1, max_value=10, value=5)
    heart_rate = st.slider('Heart Rate', min_value=60, max_value=200, value=70)
    daily_steps = st.slider('Daily Steps', min_value=1000, max_value=10000, value=8000)

    user_data = np.array([age, physical_activity, stress_level, heart_rate, daily_steps]).reshape(1, -1)

    if st.button('Predict'):
        predicted_duration = model_duration.predict(user_data)[0]
        predicted_quality = model_quality.predict(user_data)[0]

        st.write(f"Predicted Sleep Duration: {predicted_duration:.2f} hours")
        st.write(f"Predicted Sleep Quality: {predicted_quality}")

with class_tab:
    df2 = df.copy()
    df2["Gender"].replace(["Male", "Female"], [1, 0], inplace=True)
    df2["Occupation"].replace(["Software Engineer", "Doctor","Sales Representative","Software Engineer","Teacher","Nurse","Engineer","Accountant","Scientist","Lawyer","Salesperson","Manager"],  [0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
    df2["BMI Category"].replace(["Normal", "Overweight","Obese"], [0,1,2], inplace=True)
    df2["Sleep Disorder"].replace(["No Disorder", "Sleep Apnea", "Insomnia"], [0,1,2], inplace=True)
    st.markdown("## :violet[Select catagorical values from below]")
    gender = st.selectbox(' Please select  Gender:', ["Male", "Female"])
    occupation = st.selectbox(' Please select  Occupation type:', ["Software Engineer", "Doctor","Sales Representative","Software Engineer","Teacher","Nurse","Engineer","Accountant","Scientist","Lawyer","Salesperson","Manager"])
    disorder = st.selectbox(' Please select  Disorder type:', ["No Disorder", "Sleep Apnea", "Insomnia"])
    if gender == "Male":
        gen = 1
    elif gender == "Female":
        gen = 0

    if occupation == "Software Engineer": 
        occ = 0
    elif occupation  == "Doctor":
        occ = 1
    elif occupation  == "Sales Representative":
        occ = 2
    elif occupation  == "Software Engineer":
        occ = 3
    elif occupation  == "Teacher":
        occ = 4
    elif occupation  == "Nurse":
        occ = 5
    elif occupation  == "Engineer":
        occ = 6
    elif occupation  == "Accountant":
        occ = 7
    elif occupation  == "Scientist":
        occ = 8
    elif occupation  == "Lawyer":
        occ = 9
    elif occupation  == "Salesperson":
        occ = 10
    elif occupation  == "Manager":
        occ = 11

    
    if disorder == "No Disorder":
        dis = 0
    elif disorder == "Sleep Apnea":
        dis = 1
    elif disorder == "Insomnia":
        dis = 2
    st.markdown("## :violet[Select numerical values from below]")
    col1, col2, col3 = st.columns(3)

    with col1:
        age_input = st.slider("Age", 1, 100, 30, 1)
        sleep_duration_input = st.slider("Sleep Duration", 0.0, 10.0, 5.0, 0.2)

    with col2:
        quality_of_sleep = st.slider("Quality of Sleep", 2, 10, 8, 1)
        physical_input = st.slider("Physical Activity", 1, 100, 30, 1)

    with col3:
        stress_input = st.slider("Stress Level", 1, 10, 5, 1)
        heart_rate = st.slider("Heart Rate", 50, 200, 60, 1)

    daily_steps = st.slider("Daily Steps", 1000, 10000, 6000, 100)

    X = np.array(df2[["Gender", "Age", "Occupation", "Sleep Duration", "Quality of Sleep", "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps", "Sleep Disorder"]])
    y = np.array(df2["BMI Category"])

    st.markdown("## :violet[Insert the (percentage) of total data sample for test samples below]")
    test_fraction_per = st.number_input("Insert the (percentage) of total data sample for test samples", value=20, min_value=5, max_value=40, step=1)
    test_fraction = test_fraction_per / 100
    x_cat_train, x_cat_test, y_cat_train, y_cat_test = train_test_split(X, y, test_size=test_fraction, random_state=0)
    model_type = st.selectbox(
    "Select a Model",
    (
        "KNeighborsClassifier",
        "Logistic Regression",
        "SVR",
        "RandomForestClassifier",
        "DecisionTreeClassifier",
    )
)
    if model_type=="KNeighborsClassifier":
        modell = KNeighborsClassifier()
        neighbors_num=st.number_input("Specify number of neighbors to be considered", value=3, min_value=1,max_value=30, step=1)
        modell = KNeighborsClassifier(n_neighbors=neighbors_num)
        modell.fit(x_cat_train, y_cat_train)
        y_cat_pred = modell.predict(x_cat_test)
        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)
        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
        st.title("Confusion Matrix")
        plot_confusion_matrix()
        pred_res=modell.predict(np.array([[gen,age_input,occ,sleep_duration_input,quality_of_sleep,physical_input,stress_input,heart_rate,daily_steps,dis]] ))
        if pred_res== 0:
            pred_res="Normal"
        elif pred_res==1:
            pred_res= "Overweight"
        elif pred_res==2:
            pred_res= "Obese"
        st.markdown(f"### Your predicted weight is {pred_res}] based on the input values.")

    if model_type=="Logistic Regression":
        modell = LogisticRegression()
        modell.fit(x_cat_train, y_cat_train)
        sc = modell.score(x_cat_test, y_cat_test)
        y_cat_pred = modell.predict(x_cat_test)
        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)
        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
        st.title("Confusion Matrix")
        plot_confusion_matrix()
        pred_res=modell.predict(np.array([[gen,age_input,occ,sleep_duration_input,quality_of_sleep,physical_input,stress_input,heart_rate,daily_steps,dis]] ))
        if pred_res== 0:
            pred_res="Normal"
        elif pred_res==1:
            pred_res= "Overweight"
        elif pred_res==2:
            pred_res= "Obese"
        st.markdown(f"### Your predicted weight is {pred_res}] based on the input values.")
    if model_type=="Support Vector Machine":
        modell = SVR()
        modell.fit(x_cat_train, y_cat_train)
        sc = modell.score(x_cat_test, y_cat_test)
        y_cat_pred = modell.predict(x_cat_test)
        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)
        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
        st.title("Confusion Matrix")
        plot_confusion_matrix()
        pred_res=modell.predict(np.array([[gen,age_input,occ,sleep_duration_input,quality_of_sleep,physical_input,stress_input,heart_rate,daily_steps,dis]] ))
        if pred_res== 0:
            pred_res="Normal"
        elif pred_res==1:
            pred_res= "Overweight"
        elif pred_res==2:
            pred_res= "Obese"
        st.markdown(f"### Your predicted weight is {pred_res}] based on the input values.")

    if model_type=="RandomForestClassifier":
        modell = RandomForestClassifier()
        modell.fit(x_cat_train, y_cat_train)
        y_cat_pred = modell.predict(x_cat_test)
        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)
        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
        st.title("Confusion Matrix")
        plot_confusion_matrix()
        pred_res=modell.predict(np.array([[gen,age_input,occ,sleep_duration_input,quality_of_sleep,physical_input,stress_input,heart_rate,daily_steps,dis]] ))
        if pred_res== 0:
            pred_res="Normal"
        elif pred_res==1:
            pred_res= "Overweight"
        elif pred_res==2:
            pred_res= "Obese"
        st.markdown(f"### Your predicted weight is {pred_res}] based on the input values.")

    if model_type=="DecisionTreeClassifier":
        modell = DecisionTreeClassifier()
        modell.fit(x_cat_train, y_cat_train)
        sc = modell.score(x_cat_test, y_cat_test)
        y_cat_pred = modell.predict(x_cat_test)
        conf_mat = confusion_matrix(y_cat_test, y_cat_pred)
        def plot_confusion_matrix():
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(modell, x_cat_test, y_cat_test, ax=ax)
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            st.pyplot(fig)
        st.title("Confusion Matrix")
        plot_confusion_matrix()
        pred_res=modell.predict(np.array([[gen,age_input,occ,sleep_duration_input,quality_of_sleep,physical_input,stress_input,heart_rate,daily_steps,dis]] ))
        if pred_res== 0:
            pred_res="Normal"
        elif pred_res==1:
            pred_res= "Overweight"
        elif pred_res==2:
            pred_res= "Obese"
        st.markdown(f"### Your predicted weight is {pred_res}] based on the input values.")

    
