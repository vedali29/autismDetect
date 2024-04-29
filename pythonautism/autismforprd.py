import pickle

with open('autismdetection.pkl','rb') as f:
    cif = pickle.load(f)
    
prediction = model.predict(new_data)

# Print or use the prediction as needed
print('Prediction:', prediction)