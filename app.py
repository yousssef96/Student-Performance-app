from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline
from src.components.data_validation import DataValidation
from src.logger import logger

app = FastAPI(
    title="End-to-End ML project"
)

@app.get("/")
def health_check():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

class StudentData(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float


@app.post("/predict")
def predict(data: StudentData):
    try:
        df = pd.DataFrame([data.model_dump()])

        # Great Expectations validation
        validator = DataValidation(df)
        is_valid = validator.run_inference_validation()

        if not is_valid:
            raise HTTPException(
                status_code=422,
                detail="Input data failed validation — check value ranges and categories"
            )

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(df)        
        return {"prediction": result[0]}

    except HTTPException:
        raise                                        
    except Exception as e:
        logger.exception(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)