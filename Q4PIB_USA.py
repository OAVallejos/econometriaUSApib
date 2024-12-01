!pip install pandas numpy statsmodels
 import pandas as pd
 import numpy as np
 import matplotlib.pyplot as plt
 from statsmodels.tsa.arima.model import ARIMA

 # Create the DataFrame with the data provided
 data = {
     'Quarter': ['Q2 2022', 'Q3 2022', 'Q4 2022', 'Q1 2023', 'Q2 2023', 'Q3 2023', 'Q4 2023', 'Q1 2024', 'Q2 2024',  'Q3 2024'],
     'gdp %': [0.3, 2.7, 3.4, 2.8, 2.4, 4.4, 3.2, 1.6,  3.0, 2.7]
 }

 df = pd.DataFrame(data)

 # Convert 'Quarter' column to datetime type
 def convert_quarter_to_date(quarter_str):
     year = int(quarter_str[-4:])
     quarter = int(quarter_str[1])
     month = (quarter - 1) * 3 + 1
     return pd.Timestamp(year=year, month=month, day=1)

 df['Quarter'] = df['Quarter'].apply(convert_quarter_to_date)

 # Clear data (remove NaN and Inf)
 df['pib %'] = df['pib %'].replace([np.inf, -np.inf], np.nan).dropna()

 # Define a simpler ARIMA model if necessary
 model = ARIMA(df['pib  %'], order=(1, 1, 0), enforce_stationarity=False)

# Fit the model and handle possible errors
try:
model_fit = model.fit(method='statespace') # Use the statespace method
except Exception as e:  print(f"Error fitting model: {e}")
else:
# Show model summary to check fit
print(model_fit.summary())

# Forecast next quarter (Q4 2024)
forecast = model_fit.get_forecast(steps=  1)
forecast_index = pd.date_range(start='2024-10-01', periods=1, freq='QE-DEC')
forecast_value = forecast.predicted_mean.iloc[0]

# Show forecast result
print(f"Forecast  for Q4 2024:  {forecast_value:.2f}%")

# Plot forecasts alongside original data
plt.figure(figsize=(10, 5))
plt.plot(df['Quarter'], df['GDP %'], label='  Original Data', marker='o')
plt.plot(forecast_index, [forecast_value], label='Q4 2024 Forecast', marker='o', color='orange')
plt.title('GDP Forecast %'  )
plt.xlabel('Quarter')
plt.ylabel('GDP %')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot residuals
residuals =  model_fit.resid

 plt.figure(figsize=(10, 5))
 plt.plot(residuals)
 plt.title('ARIMA Model Residues')
 plt.xlabel('Index')
 plt.ylabel('Waste')
 plt.axhline(y=0, color='r', linestyle='--')
 plt.show()

 # Residue histogram
 plt.figure(figsize=(10, 5))
 plt.hist(residuals, bins=10, alpha=0.75)
 plt.title('Residue Histogram')
 plt.xlabel('Value')
 plt.ylabel('Frequency')
 plt.axvline(x=0, color='r',  linestyle='--')
 plt.show()




 SARIMAX Results                                
 ==================================================================  ============================
 Variable Dep.: GDP % No. Observations: 10
 Model: ARIMA(1, 1, 0) Log Likelihood -12.516
 Date: Sat, 30 Nov 2024 AIC 29.033
 Time: 01:52:24 BIC 29,191
 Sample: 0 HQIC 27.961
                                  - 10                                         
 Covariance Type: opg                                         
 ==================================================================  ============================
                  coef std err z P>|z|       [0.025 0.975]
 --------------------------------------------------  ----------------------------
 ar.L1 -0.1458 0.315 -0.463 0.643 -0.763 0.472
 sigma2 1.3380 0.980 1.365 0.172 -0.583 3.259
 ==================================================================  =================================
 Ljung-Box (L1) (Q): 1.25 Jarque-Bera (JB): 0.42
 Prob(Q): 0.26 Prob(JB):                          0.81
 Heteroskedasticity (H): 2.85 Skew: 0.15
 Prob(H) (two-sided): 0.41 Kurtosis: 1.93
 ==================================================================  =================================


Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
Forecast for Q4 2024: 2.74%

# Analysis of Results

1. Fitted Model: ARIMA(1, 1, 0) model, which indicates:
- p = 1: An autoregressive term.
- d = 1: The series has been differenced once to make it stationary.
- q = 0: No moving average terms have been included.

2. Log Likelihood: The log likelihood value is -12.516, which is a measure of the quality of the model fit. Higher (less negative) values ​​indicate a better fit.

3. AIC and BIC:
- AIC (Akaike Information Criterion): 29.033
- BIC (Bayesian Information Criterion): 29.191
Both criteria are used for model selection;  Lower values ​​indicate a better balance between model complexity and fit.

4. Coefficients:
- The coefficient of the autoregressive term (ar.L1) is -0.1458, suggesting that there is an inverse relationship with the previous value.
- The term `sigma2` represents the error variance and is positive, which is expected.

5. Statistical Tests:
- Ljung-Box Test: The Q value is 1.25 with a probability (Prob(Q)) of 0.26, suggesting that there is no significant autocorrelation in the residuals.
- Jarque-Bera Test: Indicates that the distribution of the residuals does not deviate significantly from normality (Prob(JB) = 0.81).

6. Forecast: The GDP forecast for Q4 2024 is 2.74%, which is a useful result for economic planning and analysis.

 # Additional Recommendations

1. Model Validation: With access to future data, we validate the model by comparing forecasts to actual values ​​as they become available.

2. Exploring Alternative Models: Consider testing other models such as SARIMA or SARIMAX if the data have seasonal components or if we wish to include exogenous variables.

3. Residual Analysis: Further analysis of the model residuals to ensure that there are no uncaptured patterns.

4. Model Fine-Tuning: Experiment with different combinations of parameters (p, d, q) and use criteria such as AIC and BIC to find the optimal model.

5. Visualization: Examine the residuals and perform additional tests to verify normality and homoscedasticity.

 # Graph residuals
 residuals = model_fit.resid

 plt.figure(figsize=(10, 5))
 plt.plot(residuals)
 plt.title('ARIMA Model Residues')
 plt.xlabel('Index')
 plt.ylabel('Waste')
 plt.axhline(y=0, color='r', linestyle='--')
 plt.show()

 # Residue histogram
 plt.figure(figsize=(10, 5))
 plt.hist(residuals, bins=10, alpha=0.75)
 plt.title('Residue Histogram')
 plt.xlabel('Value')
 plt.ylabel('Frequency')
 plt.axvline(x=0, color='r',  linestyle='--')
plt.show()