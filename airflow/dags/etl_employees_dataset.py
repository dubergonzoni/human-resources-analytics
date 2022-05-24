from datetime import datetime,date, timedelta
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from minio import Minio


DEFAULT_ARGS = {
    'owner': 'Airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 13),
}

dag = DAG('etl_employees_dataset', 
          default_args=DEFAULT_ARGS,
          schedule_interval="@once"
        )

data_lake_server = Variable.get("data_lake_server")
data_lake_login = Variable.get("data_lake_login")
data_lake_password = Variable.get("data_lake_password")

client = Minio(
        data_lake_server,
        access_key=data_lake_login,
        secret_key=data_lake_password,
        secure=False
    )

def extract():
    #build the structure for the dataframe.
    df = pd.DataFrame(data=None)
    
    #search the list of objects in the data lake.
    objects = client.list_objects('processing', recursive=True)
    
    #download each file and concatenate the empty dataframe.
    for obj in objects:
        print("Downloading file...")
        print(obj.bucket_name, obj.object_name.encode('utf-8'))

        client.fget_object(
                    obj.bucket_name,
                    obj.object_name.encode('utf-8'),
                    "/tmp/temp_.parquet",
        )
        df_temp = pd.read_parquet("/tmp/temp_.parquet")
        df = pd.concat([df,df_temp],axis=1)
    
    #persist the files to the Staging area.
    df.to_csv("/tmp/employees_dataset.csv"
               ,index=False
            )

def load():

    #load the data from Staging area.
    df_ = pd.read_csv("/tmp/employees_dataset.csv")

    #convert the data to parquet format.    
    df_.to_parquet(
            "/tmp/employees_dataset.parquet"
            ,index=False
    )

    #laod the data to the Data Lake.
    client.fput_object(
        "processing",
        "employees_dataset.parquet",
        "/tmp/employees_dataset.parquet"
    )


extract_task = PythonOperator(
    task_id='extract_data_from_datalake',
    provide_context=True,
    python_callable=extract,
    dag=dag
)

load_task = PythonOperator(
    task_id='load_file_to_data_lake',
    provide_context=True,
    python_callable=load,
    dag=dag
)

clean_task = BashOperator(
    task_id="clean_files_on_staging",
    bash_command="rm -f /tmp/*.csv;rm -f /tmp/*.json;rm -f /tmp/*.parquet;",
    dag=dag
)

extract_task >> load_task >> clean_task