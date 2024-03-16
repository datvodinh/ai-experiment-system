from fastapi import APIRouter, Response
from pymongo import MongoClient
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(
    prefix="/jobs"
)
client = MongoClient(host="mongodb://host.docker.internal", port=27017)
db = client.newdb if (
    "db" not in client.list_database_names()
) else client["db"]
job = db.job
statistic = db.statistic


class JobConfig(BaseModel):
    job_id: str
    time: datetime
    status: str


def get_config_data(job_config: JobConfig) -> dict:
    return job_config.model_dump() | {"_id": job_config.job_id}


@router.post("/{job_id}")
def add_job(job_id: str):
    if job.count_documents({"status": "running"}) == 0:
        status = "running"
    else:
        status = "queuing"
    if len(list(job.find({"job_id": job_id}))) == 0:
        job_config = JobConfig(
            job_id=job_id,
            time=datetime.now(),
            status=status
        )
        job.insert_one(get_config_data(job_config))
        return Response("Job add Completed!")
    else:
        return Response("Job already exist!")


@router.get("/{job_id}")
def get_job(job_id: str):
    if job_id == "all":
        job_dict = {}
        for j in job.find():
            j_id = j.pop("_id")
            job_dict[j_id] = j
        return job_dict
    else:
        job_find: dict = job.find_one({"job_id": job_id})
        job_find.pop("_id")
        return job_find if (
            job_find is not None
        ) else Response("Job not found!")


@router.post("/{job_id}")
def update_job(job_id: str):
    job_config: dict = job.find({"_id": job_id}).limit(1)
    if job_config["status"] == "running":
        job_config["status"] = "finish"
    else:
        job_config["status"] = "running"


@router.delete("/{job_id}")
def delete_job():
    pass
