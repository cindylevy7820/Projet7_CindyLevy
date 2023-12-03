from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Charger le modèle
pickle_in = open('model/model.pkl', 'rb')
model = pickle.load(pickle_in)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Convertir les données en dataframe
    df = pd.DataFrame(data)
    # Prétraitement des données si nécessaire
    
    # Effectuer la prédiction
    predictions = model.predict_proba(df)[:, 1]
    
    # Préparer la réponse
    response = {'predictions': predictions.tolist()}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run()
