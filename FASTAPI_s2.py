from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import joblib

# Define isSun function (must match training logic)
def isSun(hour, month):
    def approximate_sunrise_time(month):
        if month in [12, 1, 2]: return 7
        elif month in [3, 4, 5]: return 6
        elif month in [6, 7, 8]: return 5
        else: return 6

    def approximate_sunset_time(month):
        if month in [12, 1, 2]: return 17
        elif month in [3, 4, 5]: return 19
        elif month in [6, 7, 8]: return 20
        else: return 18

    return 1 if approximate_sunrise_time(month) <= hour < approximate_sunset_time(month) else 0

# Define predict_next_energy function (must match training logic)
def predict_next_energy(model, scaler, current_energy, current_humidity, current_hour, current_month):
    next_hour = (current_hour + 1) % 24
    next_month = current_month if next_hour != 0 else (current_month % 12) + 1
    
    next_is_sun = isSun(next_hour, next_month)
    
    if next_is_sun == 0:
        return 0
    
    input_data = [[current_energy, current_humidity, next_is_sun]]
    input_data_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_data_scaled)[0]
    
    return max(0, prediction)

# Load saved model and scaler (only these two are saved)
try:
    lgb_model, scaler = joblib.load('generation_prediction_model_lgb.joblib')
except FileNotFoundError:
    raise RuntimeError("The model file 'generation_prediction_model_lgb.joblib' was not found. Please train and save the model first.")

# Create FastAPI app instance
app = FastAPI()

@app.get("/api/predict_next_generation")
async def predict_next_generation(
    current_energy: float = Query(...),
    current_humidity: float = Query(...),
    current_hour: int = Query(...),
    current_month: int = Query(...)
):
    try:
        next_energy = predict_next_energy(lgb_model, scaler,
                                          current_energy,
                                          current_humidity,
                                          current_hour,
                                          current_month)
        
        return {
            "current_energy": current_energy,
            "current_humidity": current_humidity,
            "current_hour": current_hour,
            "current_month": current_month,
            "predicted_next_hour_generation": round(next_energy, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"An error occurred during prediction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)