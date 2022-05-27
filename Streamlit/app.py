#loading the packages
import pandas as pd
import streamlit as st
from minio import Minio
import urllib3
import joblib
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

#loading the files from the Data Lake
client = Minio(
        "localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

#classification model,dataset and cluster.
client.fget_object("curated","model.pkl","model.pkl")
client.fget_object("curated","dataset.csv","dataset.csv")
client.fget_object("curated","cluster.joblib","cluster.joblib")

var_model = "model"
var_model_cluster = "cluster.joblib"
var_dataset = "dataset.csv"

#loading the trained model.
model = load_model(var_model)
model_cluster = joblib.load(var_model_cluster)

#laoding the dataset.
dataset = pd.read_csv(var_dataset)

print (dataset.head())

# title
st.title("Human Resource Analytics")

# subtitle
st.markdown("This is a Data App to show the Machine Learning solution for a Human Resource Analytics problem,\
 predicting the Employee's turnover intention. Please, insert the Employee values and start the classification on the left menu.")

# print the dataset used
#st.caption("Dataframe, first 5 entries, where the model was trained:")
#st.dataframe(dataset.head())

# employees groups.
kmeans_colors = ['#91d5f9' if c == 0 else '#955555' if c == 1 else '#91f9a6' for c in model_cluster.labels_]

st.sidebar.subheader("Define the feature values for the predition of the Employee's turnover intention")

# mapping user data for each attribute
satisfaction = st.sidebar.number_input("satisfaction", value=dataset["satisfaction"].mean())
evaluation = st.sidebar.number_input("evaluation", value=dataset["evaluation"].mean())
averageMonthlyHours = st.sidebar.number_input("averageMonthlyHours", value=dataset["averageMonthlyHours"].mean())
yearsAtCompany = st.sidebar.number_input("yearsAtCompany", value=dataset["yearsAtCompany"].mean())

# inserting a botton in the screen
btn_predict = st.sidebar.button("Start the Classification")

# verifying if the botton was added
if btn_predict:
    data_test = pd.DataFrame()
    data_test["satisfaction"] = [satisfaction]
    data_test["evaluation"] =	[evaluation]    
    data_test["averageMonthlyHours"] = [averageMonthlyHours]
    data_test["yearsAtCompany"] = [yearsAtCompany]
    
    #print the data from the test    
    print(data_test)

    #performs the prediction
    result = predict_model(model, data=data_test)
    intention = int(predict_model(model, data=data_test).iloc[0][4])
    if intention == 1:
        label = 'leave'
    if intention == 0:
        label = 'stay'
    prob = (predict_model(model, data=data_test).iloc[0][5] * 100).round(2)
    
    #st.subheader("Employee's turnover intention:")
    #st.text(result)
    st.text('')
    st.text('')
    st.text('')


    st.subheader('Result:')
    st.write(result)
    st.caption("Label 1 = Turnover  |  Label 0 = Retention")
    #font_size = st.slider("Enter a font size", 1, 300, value=30)
    html_str = f"""<style>p.a {{font: bold {20}px Courier;}}
    </style><p class="a">The model predicted this Employee to <span style="color: #999999">{label}</span>, with a probability of {prob}%.</p>"""
    st.markdown(html_str, unsafe_allow_html=True)

    #st.write(result)
    st.subheader('Cluster Analysis')

    fig = plt.figure(figsize=(10, 6))
    plt.scatter( x="satisfaction"
                ,y="evaluation"
                ,data=dataset[dataset.turnover==1],
                alpha=0.25,color = kmeans_colors)

    plt.xlabel("Satisfaction")
    plt.ylabel("Evaluation")

    plt.scatter( x=model_cluster.cluster_centers_[:,0]
                ,y=model_cluster.cluster_centers_[:,1]
                ,color="black"
                ,marker="X",s=40)
    
    plt.scatter( x=[satisfaction]
                ,y=[evaluation]
                ,color="grey"
                ,marker="X",s=500)

    plt.title("Employees Groups - Satisfection vs Evaluation.")
    plt.show()
    st.pyplot(fig) 

    employee_X = '<p style="font-family:sans-serif; color:Grey; font-size: 25px;">X = Employee classification.</p>'
    st.caption(employee_X, unsafe_allow_html=True)
    group_blue = '<p style="font-family:sans-serif; color: #91f9a6 ; font-size: 15px;">High Evaluation and High Satisfaction</p>'
    st.caption(group_blue, unsafe_allow_html=True)
    group_green = '<p style="font-family:sans-serif; color: #91d5f9 ; font-size: 15px;">Low Evaluation and Intermediary Satisfaction</p>'
    st.caption(group_green, unsafe_allow_html=True)
    group_red = '<p style="font-family:sans-serif; color:#955555; font-size: 15px;">High Evaluation and Low Satisfaction</p>'
    st.caption(group_red, unsafe_allow_html=True)