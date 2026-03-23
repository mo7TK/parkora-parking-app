from fastapi import FastAPI

from routes.spots import router as spots_router

app = FastAPI()

app.include_router(spots_router)


@app.get("/")
def health_check():
    return {"message": "Parkora backend is running"}