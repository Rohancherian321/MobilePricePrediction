from django.shortcuts import render
import pickle
# Create your views here.
import joblib

# Load the model

model = joblib.load('model_randmo2.pkl')

import numpy as np

max_values = np.array([1998.0, 64.0, 200.0, 8.0,  1960.0, 1998.0, 3998.0, 19.0, 18.0])
min_values = np.array([501.0, 2.0, 0.1, 80.0, 0.0, 500.0, 256.0, 5.0, 0.0])

from django.shortcuts import render


def predict_view(request):
    if request.method == 'POST':
        # Extract the input data from request.POST
        battery_power = float(request.POST.get('battery_power'))
       
        
      
      
      
        int_memory = int(request.POST.get('int_memory'))
        
        mobile_wt = int(request.POST.get('mobile_wt'))
        n_cores = int(request.POST.get('n_cores'))
  
        px_height = int(request.POST.get('px_height'))
        px_width = int(request.POST.get('px_width'))
        ram = int(request.POST.get('ram'))
        sc_h = int(request.POST.get('sc_h'))
        sc_w = int(request.POST.get('sc_w'))
        

        # Create a list with the input data
        input_data = [
            battery_power,  int_memory, mobile_wt, 
            n_cores, px_height, px_width, ram, sc_h, sc_w
        ]

        
# Convert the input data to a NumPy array
        new_data = np.array(input_data)

        # Scale the new data using the same Min-Max scaling as your training data
        new_data_scaled = (new_data - min_values) / (max_values - min_values)

        # Reshape the scaled new data to match the shape expected by the model
        new_data_scaled = new_data_scaled.reshape(1, -1)

        # Use the `.predict()` method to make a prediction on the new data
        predicted_price_range = model.predict(new_data_scaled)

        # Map the predicted values to their corresponding labels
        price_range_labels = {
            0: "Low Cost",
            1: "Medium Cost",
            2: "High Cost",
            3: "Very High Cost"
        }

        # Get the predicted label
        predicted_label = price_range_labels[predicted_price_range[0]]


        return render(request, 'prediction_result.html', {'predicted_label': predicted_label})
    
    return render(request, 'predict.html')



# Input data
