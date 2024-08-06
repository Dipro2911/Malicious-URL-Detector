from selenium import webdriver 
from selenium.webdriver.chrome.options import Options 
from selenium.webdriver.common.by import By 
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from datetime import datetime
import psutil
import pygetwindow as gw
import sys

def is_chrome_running():
    for process in psutil.process_iter(['pid', 'name']):
        if 'chrome.exe' in process.info['name']:
            return True
    return False

flag = 0

captured_url = "captured_url.csv"
output_file = "predicted.csv"

# Load the saved model and tokenizer
loaded_model = load_model('url_detection.h5')
tokenizer = Tokenizer()
tokenizer.word_index = np.load('url_check.npy', allow_pickle=True).item()

predictions_list=[]

driver = webdriver.Chrome()
driver.maximize_window()
url_default = driver.get("https://www.google.com")
prev = url_default
while True:
    try:        
        if not is_chrome_running():
        # If Chrome is not running, close the program
            flag = 1
            break

        url = driver.current_url
        if url != prev:
            print(url)
            prev = url
        
            existing_data = pd.DataFrame()
            new_data = pd.DataFrame({'URL': [url]})
            existing_data = pd.concat([existing_data, new_data], ignore_index=True)
            existing_data.to_csv(captured_url, index=False)
            
            df = pd.read_csv('captured_url.csv')
            urls = df['URL'].tolist()

            max_len = 100  # Adjust based on your model's input length
            url_sequence = tokenizer.texts_to_sequences([url])
            url_padded = pad_sequences(url_sequence, maxlen=max_len)

            # Make prediction
            prediction = loaded_model.predict(url_padded)

            # Decode the prediction (assuming it's a classification task)
            predicted_label = np.argmax(prediction)
            if predicted_label == 0:
                label = 'benign'
            elif predicted_label == 1:
                label = 'malicious'
            
            current_datetime = datetime.now()
            time = current_datetime.strftime("%H:%M:%S")       
            date = current_datetime.date()        

    
            predictions_list.append({'URL': url, 'Prediction': label, 'Date': date, 'Time': time})

            try:
                existing_data = pd.read_csv(output_file)
            except FileNotFoundError:
                existing_data = pd.DataFrame()

            # Append the new predictions to the existing data
            output_data = pd.concat([existing_data, pd.DataFrame(predictions_list)], ignore_index=True)

            # Write the DataFrame to the CSV file
            output_data.to_csv(output_file, index=False)


    except:
        predictions_list.append({'URL': url, 'Prediction': label, 'Date': date, 'Time': time})
        try:
            existing_data = pd.read_csv(output_file)
        except FileNotFoundError:
            existing_data = pd.DataFrame()

        # Append the new predictions to the existing data
        output_data = pd.concat([existing_data, pd.DataFrame(predictions_list)], ignore_index=True)

        # Write the DataFrame to the CSV file
        output_data.to_csv(output_file, index=False)


    


    
if flag == 1:
    print("Exiting")
    sys.exit()