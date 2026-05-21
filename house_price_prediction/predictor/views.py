from pathlib import Path

import joblib
from django.http import HttpResponseBadRequest
from django.shortcuts import render

MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "model" / "house_model.pkl"
)
model = joblib.load(MODEL_PATH)


import pandas as pd

def predict_price(request):
    if request.method == "GET":
        return render(request, "result.html")

    if request.method != "POST":
        return HttpResponseBadRequest("Unsupported request method.")

    field_names = [
        "income",
        "age",
        "rooms",
        "bedrooms",
        "population",
        "occupancy",
        "latitude",
        "longitude",
    ]

    try:
        features_dict = {name: float(request.POST[name]) for name in field_names}
    except (KeyError, TypeError, ValueError):
        return render(
            request,
            "result.html",
            {"error": "Please fill in all fields with valid numeric values."},
            status=400,
        )
        
    # User inputs actual income, so we divide by 10,000 to match dataset scale
    features_dict["income"] = features_dict["income"] / 10000.0

    # Create a DataFrame to pass to the model, so it has feature names
    input_df = pd.DataFrame([{
        "MedInc": features_dict["income"],
        "HouseAge": features_dict["age"],
        "AveRooms": features_dict["rooms"],
        "AveBedrms": features_dict["bedrooms"],
        "Population": features_dict["population"],
        "AveOccup": features_dict["occupancy"],
        "Latitude": features_dict["latitude"],
        "Longitude": features_dict["longitude"]
    }])

    prediction = model.predict(input_df)
    # The California housing dataset target is in $100,000s
    price_val = prediction[0] * 100000
    if price_val < 0:
        price_val = 0 # Prevent negative prices
        
    formatted_price = f"${price_val:,.2f}"
    
    # Calculate gauge percentage based on a max scale of $1,000,000
    max_price_scale = 1000000.0
    gauge_percentage = min((price_val / max_price_scale) * 100, 100)

    return render(request, "result.html", {
        "price": formatted_price,
        "raw_price": price_val,
        "gauge_percentage": gauge_percentage
    })
