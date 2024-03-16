from fastapi import APIRouter
from pymongo import MongoClient
from pydantic import BaseModel
from datetime import datetime


router = APIRouter(
    prefix="/stat"
)
client = MongoClient(host="mongodb://host.docker.internal", port=27017)
db = client.newdb if (
    "db" not in client.list_database_names()
) else client["db"]
statistic = db.statistic


class Statistic(BaseModel):
    time: datetime
    job_id: str
    best_config: dict[str, str]
    all_config: list[dict[str, str]]


@router.get("/{job_id}")
def get_statistic(job_id: str):
    pass


@router.post("/")
def update_statistic(job_id: str):
    pass
