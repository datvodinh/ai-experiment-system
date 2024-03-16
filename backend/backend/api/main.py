from fastapi import FastAPI
import server
# FASTAPI
app = FastAPI()
app.include_router(server.job_router)
app.include_router(server.stat_router)
