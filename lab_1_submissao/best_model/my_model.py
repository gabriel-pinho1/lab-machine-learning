import joblib


def predict(new_data):
    
    model = joblib.load("model.pkl")
    predictions = model.predict(new_data)
    
    return predictions