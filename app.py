import streamlit as st
import datetime
import pandas as pd
import joblib

today = datetime.date.today()


@st.cache_resource
def load_model():
    model_path = "LogisticRegression.pkl"
    return joblib.load(model_path)

def get_clinical_explanation(pipeline, patient_data):
    """
    Refactored version of your explanation logic
    returns data instead of printing.
    """
    # 1. Get Probability
    prob = pipeline.predict_proba(patient_data)[0][1]

    # 2. Extract components
    classifier = pipeline.named_steps['classifier']
    selector = pipeline.named_steps['selector']
    preprocessor = pipeline.named_steps['preprocessor']

    # 3. Transform data
    transformed_data = preprocessor.transform(patient_data)
    selected_data = selector.transform(transformed_data)

    # 4. Calculate Contribution
    contributions = classifier.coef_[0] * selected_data[0]
    all_names = preprocessor.get_feature_names_out()
    selected_names = all_names[selector.get_support()]

    # 5. Map and Sort
    importance = pd.Series(contributions, index=selected_names)
    top_drivers = importance.abs().sort_values(ascending=False).head(3)

    drivers_list = []
    for name, val in top_drivers.items():
        clean_name = name.split('__')[-1]
        direction = "🔴 Increasing risk" if importance[name] > 0 else "🟢 Decreasing risk"
        drivers_list.append(f"**{clean_name}**: {direction}")

    # Risk Level logic
    risk_level = "High" if prob > 0.8 else ("Moderate" if prob > 0.5 else "Low")

    return {
        "prob": prob,
        "risk_level": risk_level,
        "drivers": drivers_list,
        "features_count": selector.k
    }


model = load_model()

st.title("Heart Disease Prediction App")
st.write("Enter patient information below:")

age = st.number_input(label="Age",value=None)
gender = st.selectbox("Gender", [0, 1],index=None, format_func=lambda x: "Female" if x == 0 else "Male",placeholder="Choose your Gender")
cp = st.selectbox("Chest Pain Type (cp)", [1, 2, 3,4],index=None,placeholder="Choose Chest Pain Type (cp)")
bp = st.number_input("Blood Pressure", 30, 260, None,placeholder="Enter resting blood pressure (mm Hg)")
chol = st.number_input("Cholesterol", 100, 500, None,placeholder="Enter cholesterol (mg/dl)")
fbs = st.number_input("Fasting Blood Sugar",min_value=20,max_value=600,value=None,placeholder="Enter fasting blood sugar (mg/dl)")
ekg = st.selectbox("EKG", [0, 1, 2],index=None,placeholder="EKG Results")
hr = st.number_input("Max Heart Rate Achieved", 60, 220, None,placeholder="Enter max heart rate achieved")
exang = st.checkbox("Exercise Induced Angina")
stdep = st.number_input("ST depression", 0.0, 6.0, None,placeholder="Enter ST depression induced by exercise relative to rest")
slope = st.selectbox("Slope", [1, 2,3], index=None,placeholder="Choose the slope of the peak exercise ST segment")
fluro = st.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3],index = None,placeholder="Choose number of major vessels (0-3) colored by flourosopy")
thal = st.selectbox("Thal", [3,6,7],index=None,placeholder="Choose thal type")
work = st.selectbox("Work Type", ["Private","children","Self-employed","Govt_job"],index=None,placeholder="Enter work type")
smoke = st.selectbox("Smoking Status", ["smokes","never smoked","formerly smoked","Unknown"],index=None,placeholder="Enter smoking status")


if fbs is not None:
    fbs = 1 if fbs > 120 else 0

missing_fields = []

if age is None:
    missing_fields.append("Birth date")

if gender is None:
    missing_fields.append("Gender")

if cp is None:
    missing_fields.append("Chest Pain Type")

if bp is None:
    missing_fields.append("Resting Blood Pressure")

if chol is None:
    missing_fields.append("Cholesterol")

if fbs is None:
    missing_fields.append("Fasting Blood Sugar")

if ekg is None:
    missing_fields.append("EKG")

if hr is None:
    missing_fields.append("Max Heart Rate Achieved")

if stdep is None:
    missing_fields.append("ST")

if slope is None:
    missing_fields.append("Slope")

if fluro is None:
    missing_fields.append("Number of Major Vessels")

if thal is None:
    missing_fields.append("Thal")

if work is None:
    missing_fields.append("Work Type")

if smoke is None:
    missing_fields.append("Smoking Status")

predict_disabled = len(missing_fields) > 0

if predict_disabled:
    st.warning(
        "Please complete the following fields:\n\n- " +
        "\n- ".join(missing_fields)
    )

if st.button("Predict", disabled=predict_disabled):
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Chest pain type": cp,
        "BP": bp,
        "Cholesterol": chol,
        "FBS over 120": fbs,
        "EKG results": ekg,
        "Max HR": hr,
        "Exercise angina": int(exang),
        "ST depression": stdep,
        "Slope of ST": slope,
        "Number of vessels fluro": fluro,
        "Thallium": thal,
        "work_type": work,
        "smoking_status": smoke,
    }])
    # Get explanation data
    results = get_clinical_explanation(model, input_data)

    st.divider()
    st.subheader("Clinical Summary")

    # Display Risk with Metrics
    cols = st.columns(2)
    cols[0].metric("Risk Assessment", results["risk_level"])
    cols[1].metric("Disease Probability", f"{results['prob']:.1%}")

    # Display Drivers
    st.write("### Top Diagnostic Drivers")
    for driver in results["drivers"]:
        st.write(driver)

    # Footer Info
    st.info(f"Model Integrity: Optimized via {results['features_count']} features.")

    # Validation Recall (if available)
    val_score = getattr(model, 'val_score_', None)
    if val_score:
        st.caption(f"Historical Validation Recall: {val_score:.2%}")

