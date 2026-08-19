from flask import Flask, render_template, request
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")

    try:
        # Extract and cast numeric inputs safely with default fallback values
        data = CustomData(
            gender=request.form.get("gender"),
            SeniorCitizen=int(request.form.get("SeniorCitizen", 0)),
            Partner=request.form.get("Partner"),
            Dependents=request.form.get("Dependents"),
            tenure=int(request.form.get("tenure", 0)),
            PhoneService=request.form.get("PhoneService"),
            MultipleLines=request.form.get("MultipleLines"),
            InternetService=request.form.get("InternetService"),
            OnlineSecurity=request.form.get("OnlineSecurity"),
            OnlineBackup=request.form.get("OnlineBackup"),
            DeviceProtection=request.form.get("DeviceProtection"),
            TechSupport=request.form.get("TechSupport"),
            StreamingTV=request.form.get("StreamingTV"),
            StreamingMovies=request.form.get("StreamingMovies"),
            Contract=request.form.get("Contract"),
            PaperlessBilling=request.form.get("PaperlessBilling"),
            PaymentMethod=request.form.get("PaymentMethod"),
            MonthlyCharges=float(request.form.get("MonthlyCharges", 0.0)),
            TotalCharges=float(request.form.get("TotalCharges", 0.0)),
        )

        pred_df = data.get_data_as_data_frame()

        pipeline = PredictPipeline()
        results = pipeline.predict(pred_df)

        return render_template("home.html", results=results[0])

    except ValueError as e:
        # Catch type conversion errors gracefully
        return render_template(
            "home.html", error="Please ensure all numerical fields are filled with valid numbers."
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0")