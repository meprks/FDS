import streamlit as st
import pandas as pd  
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title("Here is the story, I want to show") 

df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")
show_story = st.radio("Do you want to see the story?", ("Yes", "No"))
if show_story == "Yes":
    st.header("Choose Plot")
    plot_type = st.selectbox(
    "Select plot",
    ( "Age vs. Sleep Duration", 
     "Physical Activity Level vs. Sleep Duration", "Stress Level vs. Sleep Duration",
     "BMI Category vs. Sleep Duration",
     "Sleep Disorder vs. Sleep Duration")
)


    if plot_type == "Age vs. Sleep Duration":
    # Create a joint plot based on the selected hue category
       joint_plot = sns.jointplot(data=df, x="Age", y="Sleep Duration", hue="Gender", kind="scatter")
       st.pyplot()
       st.write("### What does the Plot show?")
       st.write(" The plot visualizes the relationship between Age and Sleep Duration. Based upon Gender, Female have with high age group has higher sleep duration than male. Aslo, the average sleep duration for male lies between 7 and 8 whereas the average sleep duration for female is 8 to 9.")
  

    elif plot_type == "Physical Activity Level vs. Sleep Duration":
        st.header("Physical Activity Level vs. Sleep Duration")
        sns.regplot(data=df, x="Physical Activity Level", y="Sleep Duration")
        st.pyplot()
        st.write("### What does the Plot show?")
        st.write(" The plot visualizes the relationship between Physical activity Level and Sleep Duration. The plot shows the positive correlation between the data. This indicate that person with high physical activity level has high sleep duration.")

    elif plot_type == "Stress Level vs. Sleep Duration":
        st.header("Stress Level vs. Sleep Duration")
        sns.boxplot(data=df, x="Stress Level", y="Sleep Duration")
        st.pyplot()
        st.write("### What does the Plot show?")
        st.write(" The plot visualizes the relationship between stress level and Sleep Duration. The plot shows that person with stress level has high sleep duration indicating stress level directly affect the sleep duration.")

    elif plot_type == "BMI Category vs. Sleep Duration":
        st.header("BMI Category vs. Sleep Duration")
        sns.boxplot(data=df, x="BMI Category", y="Sleep Duration")
        st.pyplot()
        st.write("### What does the Plot show?")
        st.write(" The plot visualizes the relationship between BMI and Sleep Duration. The plot shows that the person with the normal weight have better sleep duration than the person with overweight or obese.")

    elif plot_type == "Sleep Disorder vs. Sleep Duration":
    
        st.header("Sleep Disorder vs. Sleep Duration")
        sns.boxplot(data=df, x="Sleep Disorder", y="Sleep Duration")
        st.pyplot()
        st.write("### What does the Plot show?")
        st.write(" The plot visualizes the relationship between sleep disorder and Sleep Duration. The plot shows the person with insomnia has lower sleep duration.")


