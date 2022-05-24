from datetime import datetime,date, timedelta
import math
import pandas as pd
from io import BytesIO
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from minio import Minio
from sqlalchemy.engine import create_engine


DEFAULT_ARGS = {
    'owner': 'Airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 13),
}

dag = DAG('etl_time_in_company_att', 
          default_args=DEFAULT_ARGS,
          schedule_interval="@once"
        )

data_lake_server = Variable.get("data_lake_server")
data_lake_login = Variable.get("data_lake_login")
data_lake_password = Variable.get("data_lake_password")

database_server = Variable.get("database_server")
database_login = Variable.get("database_login")
database_password = Variable.get("database_password")
database_name = Variable.get("database_name")


url_connection = "mysql+pymysql://{}:{}@{}/{}".format(
                 str(database_login)
                ,str(database_password)
                ,str(database_server)
                ,str(database_name)
                )

engine = create_engine(url_connection)

client = Minio(
        data_lake_server,
        access_key=data_lake_login,
        secret_key=data_lake_password,
        secure=False
    )

def extract():

    #query to consult the data.
    query = """SELECT hire_date
            FROM employees;"""

    df_ = pd.read_sql_query(query,engine)
    
    #persist the files in the Staging area.
    df_.to_csv( "/tmp/time_in_company.csv"
                ,index=False
            )

def transform():

    #load the data from Staging area.
    df_ = pd.read_csv("/tmp/time_in_company.csv")

    #convert data to datetime format.
    df_["hire_date"] = pd.to_datetime(df_["hire_date"])

    #define the reference date.
    date_referencia = date(2021, 1, 1)

    #calculate the diference fo days between the hiring date and the reference date.
    days_diff = []
    for d in df_["hire_date"]:
        diff = date_referencia - d.date()
        days_diff.append(diff.days)

    #convert to integer format.
    nyears = []
    for ndays in days_diff:
        nyears.append(int(math.ceil(ndays / 365)))
    
    #assign the data to the temporary dataframe.
    df_["time_in_company"] = nyears

    #persit the files to the Staging area.
    df_[["time_in_company"]].to_csv( "/tmp/time_in_company.csv"
                ,index=False
            )
    
def load():

    #load the data from Staging area.
    df_ = pd.read_csv("/tmp/time_in_company.csv")

    #convert data to parquet format.    
    df_.to_parquet(
            "/tmp/time_in_company.parquet"
            ,index=False
    )

    #load the data to the Data Lake.
    client.fput_object(
        "processing",
        "time_in_company.parquet",
        "/tmp/time_in_company.parquet"
    )


extract_task = PythonOperator(
    task_id='extract_data_from_database',
    provide_context=True,
    python_callable=extract,
    dag=dag
)

transform_task = PythonOperator(
    task_id='transform_data',
    provide_context=True,
    python_callable=transform,
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

extract_task >> transform_task >> load_task >> clean_task