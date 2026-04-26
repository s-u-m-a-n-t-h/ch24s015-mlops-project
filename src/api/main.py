from fastapi import FastAPI

app = FastAPI(title="Portfolio Allocation API")

@app.get("/")
def read_root():
    return {"message": "Portfolio Allocation API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}
