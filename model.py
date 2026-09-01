import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PATH = "medical.csv"
TARGET = "charges"
MODEL_PATH = "best_hospital_bill_model.joblib"
RANDOM_STATE = 42

NUMERIC_FEATURES = ["age", "bmi", "children"]
CATEGORICAL_FEATURES = ["sex", "smoker", "region"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def build_preprocessor():
    ohe_params = {"handle_unknown": "ignore", "drop": "first"}
    try:
        ohe = OneHotEncoder(sparse_output=False, **ohe_params)
    except TypeError:
        ohe = OneHotEncoder(sparse=False, **ohe_params)

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", ohe, CATEGORICAL_FEATURES),
        ]
    )

def calculate_margin_accuracy(y_true, y_pred, tolerance=0.15):
    """Calculates % of predictions within a percentage margin of the actual bill."""
    diff = np.abs((y_true - y_pred) / y_true)
    return np.mean(diff <= tolerance) * 100

def train_and_save_best_model(df):
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor()
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=RANDOM_STATE)
    }

    best_margin_acc = 0  
    best_pipe = None
    best_name = ""

    print("\n" + "="*85)
    # Highlighted Accuracy Columns
    print(f"{'Model Name':<20} | {'R2 Score %':<12} | {'Margin Acc (±15%)':<20}")
    print("-" * 85)

    for name, model in models.items():
        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        
        r2 = r2_score(y_test, preds) * 100
        margin_acc = calculate_margin_accuracy(y_test, preds, tolerance=0.15)
        
        print(f"{name:20} | {r2:10.2f}% | {margin_acc:18.2f}%")

        # Select the best based on practical Margin Accuracy
        if margin_acc > best_margin_acc:
            best_margin_acc = margin_acc
            best_pipe = pipe
            best_name = name

    print("-" * 85)
    print(f"WINNING MODEL: {best_name}")
    print(f"Final Prediction Accuracy: {best_margin_acc:.2f}% (within 15% of actual value)")
    print("="*85 + "\n")

    joblib.dump(best_pipe, MODEL_PATH)
    return best_pipe

def predict_interactive(model):
    print("\n--- Patient Details for Prediction ---")
    data = {
        "age": get_input("Age: ", int, 0, 120),
        "bmi": get_input("BMI: ", float, 10.0, 70.0),
        "children": get_input("Number of Children: ", int, 0, 20),
        "sex": get_input("Sex (male/female): ", options=["male", "female"]),
        "smoker": get_input("Smoker (yes/no): ", options=["yes", "no"]),
        "region": get_input("Region (northeast/northwest/southeast/southwest): ", 
        options=["northeast", "northwest", "southeast", "southwest"])
    }

    sample_df = pd.DataFrame([data])
    prediction = model.predict(sample_df)[0]
    
    lower_bound = prediction * 0.85
    upper_bound = prediction * 1.15

    print(f"\n" + "*"*40)
    print(f"PREDICTED BILL: ${prediction:,.2f}")
    print(f"ACCURACY RANGE (15%): ${lower_bound:,.2f} - ${upper_bound:,.2f}")
    print("*"*40)

def get_input(prompt, type_=str, min_=None, max_=None, options=None):
    while True:
        try:
            val = input(prompt).strip()
            if options and val.lower() not in [o.lower() for o in options]:
                print(f"Invalid choice. Choose from: {', '.join(options)}")
                continue
            
            typed_val = type_(val)
            if min_ is not None and typed_val < min_:
                print(f"Value must be at least {min_}.")
                continue
            if max_ is not None and typed_val > max_:
                print(f"Value must be no more than {max_}.")
                continue
            return typed_val
        except ValueError:
            print(f"Invalid input. Please enter a {type_.__name__}.")

def predict_interactive(model):
    print("\n--- Patient Details for Prediction ---")
    data = {
        "age": get_input("Age: ", int, 0, 120),
        "bmi": get_input("BMI: ", float, 10.0, 70.0),
        "children": get_input("Number of Children: ", int, 0, 20),
        "sex": get_input("Sex (male/female): ", options=["male", "female"]),
        "smoker": get_input("Smoker (yes/no): ", options=["yes", "no"]),
        "region": get_input("Region (northeast/northwest/southeast/southwest): ", 
        options=["northeast", "northwest", "southeast", "southwest"])
    }

    sample_df = pd.DataFrame([data])
    prediction = model.predict(sample_df)[0]
    print(f"\n>>> Predicted Hospital Bill: ${prediction:,.2f} <<<")

def main():
    if not os.path.exists(PATH):
        print(f"Error: Dataset '{PATH}' not found.")
        return

    df = pd.read_csv(PATH)
    
    if os.path.exists(MODEL_PATH):
        print(f"Existing model found. Loading...")
        model = joblib.load(MODEL_PATH)
    else:
        model = train_and_save_best_model(df)

    while True:
        predict_interactive(model)
        if get_input("\nPredict another patient? (y/n): ", options=["y", "n"]).lower() == 'n':
            break

if __name__ == "__main__":
    main()