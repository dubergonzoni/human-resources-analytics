#loading the packages
import pandas as pd
import streamlit as st
from minio import Minio
import joblib
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

#loading the files from the Data Lake
client = Minio(
        "localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
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
st.markdown("This is a Data App to show the Machine Learning solution for the Human Resource Analytics problem.")

# print the dataset used
st.dataframe(dataset.head())

# employees groups.
kmeans_colors = ['green' if c == 0 else 'red' if c == 1 else 'blue' for c in model_cluster.labels_]

st.sidebar.subheader("Defina os atributos do empregado para predição de turnover")

# mapping user data for each attribute
satisfaction = st.sidebar.number_input("satisfaction", value=dataset["satisfaction"].mean())
evaluation = st.sidebar.number_input("evaluation", value=dataset["evaluation"].mean())
averageMonthlyHours = st.sidebar.number_input("averageMonthlyHours", value=dataset["averageMonthlyHours"].mean())
yearsAtCompany = st.sidebar.number_input("yearsAtCompany", value=dataset["yearsAtCompany"].mean())

# inserting a botton in the screen
btn_predict = st.sidebar.button("Realizar Classificação")

# verifying if the botton was added
if btn_predict:
    data_teste = pd.DataFrame()
    data_teste["satisfaction"] = [satisfaction]
    data_teste["evaluation"] =	[evaluation]    
    data_teste["averageMonthlyHours"] = [averageMonthlyHours]
    data_teste["yearsAtCompany"] = [yearsAtCompany]
    
    #print the data from the test    
    print(data_teste)

    #performs the prediction
    result = predict_model(model, data=data_teste)
    
    st.write(result)



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
                ,marker="X",s=100)
    
    plt.scatter( x=[satisfaction]
                ,y=[evaluation]
                ,color="yellow"
                ,marker="X",s=300)

    plt.title("Employees Groups - Satisfection vs Evaluation.")
    plt.show()
    st.pyplot(fig) 