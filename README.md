# Inhouse-Temperature-prediction-using-Python  
Using ML models and temperature data to predict inhouse temperature  
This repo contains python files to predict inhouse temperature using weather data and scikit-learn ML models  
1) sklearn_kfold.py uses 2 years of measured temperature data of a bathroom, to predict the future temperature of this bathrooom.    
2) sklearn_kfold_weather_data.py adds 2 years of historical weather data (temperature, pressure, humidity) of the area the house is a located, to improve the predictive performance of the algorithms.  
3) sklearn_tuning.py performs hyperaparameter tuning on each of the algorithms tested, to further improve their performance. This can be used with either the bathroom data or the additional weather data.
