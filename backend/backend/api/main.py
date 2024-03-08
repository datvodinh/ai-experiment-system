from fastapi import FastAPI
from pymongo import MongoClient

app = FastAPI()
client = MongoClient(host="mongodb://host.docker.internal", port=27017)
db_list = client.list_database_names()

if "db" in db_list:
    print("Database have already initialized!")
else:
    print("Initalizing Database!")
    db = client.newdb

experiment = db.experiment
experiment.insert_one(
    {
        "name": "Vo Dinh Dat",
        "age": 22,
        "university": "HUST"
    }
)

@app.get("/")
def hello():
    return {"Hello": "World!"}


@app.get("/info")
def get_info():
    users = [u for u in experiment.find()]
    p1 = users[0]
    p1.pop("_id")
    return p1
