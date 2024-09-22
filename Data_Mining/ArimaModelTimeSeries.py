import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

np.random.seed(0)
dates = pd.date_range(start='2020-01-01', periods=36, freq='M')
sales = np.random.randint(100, 200, size=len(dates))
data = pd.DataFrame({'Month': dates, 'Sales': sales})
data.set_index('Month', inplace=True)

data.plot()
plt.title('Monthly Sales Data')
plt.show()

model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()

print(model_fit.summary())

forecast = model_fit.forecast(steps=12)
print(forecast)

plt.figure()
plt.plot(data, label='Original')
plt.plot(forecast, label='Forecast', color='red')
plt.title('Sales Forecast')
plt.legend()
plt.show()








# --------------------------------------Using Real Datasets---------------------------------------
# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.arima.model import ARIMA

# url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
# data = pd.read_csv(url, parse_dates=['Month'], index_col='Month')

# data.plot()
# plt.show()

# model = ARIMA(data, order=(5, 1, 0))
# model_fit = model.fit()

# print(model_fit.summary())

# forecast = model_fit.forecast(steps=12)

# data.plot()
# forecast.plot()
# plt.show()