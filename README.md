# Wind-Turbine-Energy-Prediction-Model
## Objective
- Create an energy model using a multiple linear regression function, Random forest and XGBoost regressor function to predict the energy generation of a wind turbine based on 2018 Scada Data of a Wind Turbine in Turkey. 
- Visualise the dataset and results. 
- Compare the two models.
## About Dataset
### Context:

-In Wind Turbines, Scada Systems measure and save data's like wind speed, wind direction, generated power etc. for 10 minutes intervals. This file was taken from a wind turbine's scada system that is working and generating power in Turkey.

### Content:

The data's in the file are:

- Date/Time (for 10 minutes intervals).

- LV ActivePower (kW): The power generated by the turbine for that moment.

- Wind Speed (m/s): The wind speed at the hub height of the turbine (the wind speed that turbine use for electricity generation).

- Theoretical_Power_Curve (KWh): The theoretical power values that the turbine generates with that wind speed which is given by the turbine manufacturer.

- Wind Direction (°): The wind direction at the hub height of the turbine (wind turbines turn to this direction automaticly).

### Libraries Used: Numpy, Pandas, Matplotlib, Seaborn, LinearRegression, RandomForestRegressor XGB Regressor from Sklearn.

#### In this project, three different models are built to predict the energy generation of wind turbine. The project consisted in different steps. First, imported the necessary libraries, then loaded and cleaned the data and prepared it for modelling. Then, splitted the data into training and test sets, built the model, and evaluated its performance on the test set using mean squared error and R-squared metrics.
#### Here, XGBoost and Random Forest Model is reliable and effective model to predict the output with high accuracy in compare with Multiple Linear Regression model.
