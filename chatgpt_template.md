Create a early machine learning ipynb jupyter notebook and associated csv file. First, be creative and describe a creative dataset based on something from a video game(mmorpg, call of duty stats, etc), cars stats, or a particular movie series that follows the following format: 

It starts out with imports. here are the ones our class has gone over:
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression```

then by reading the data in with either `pd.read_csv("data.csv")` or `np.genfromtxt('data.csv', delimiter=',', skip_header=1)` depending on if there is a header 

Then we shuffle the data with `data = data.sample(frac=1, random_state=142).reset_index(drop=True)`

We usually have this helper function:
```
def return_percentage(value: float, perc: float) -> int:
    return round(value * perc)
```

Then we split the data into training and testing data:
```
test_data = data_randomized[:return_percentage(len(data_randomized), 0.05)] # 5% of the data
training_data = data_randomized[return_percentage(len(data_randomized), 0.05):] # 95% of the data
```

Then we split the data into x(features) and y(target):
```y_train = train_data.iloc[:, 60]
x_train = train_data.iloc[:, :60]

y_test = test_data.iloc[:, 60]
x_test = test_data.iloc[:, :60]
```

Then we train the model:
```
model = LinearRegression() # or CategoricalNB() or GaussianNB()
model.fit(x_train, y_train)
```

Then we test the model:
```model.score(x_test, y_test)```

Then we can make predictions:
```model.predict(x_test)```

Then we can plot the data:
```
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, model.predict(x_test), color='blue')
plt.title('Title')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
```

Sometimes there will be print statements that show the .coef_ or .intercept_ of the model.


Importantly, you should make the jupyter notebook and csv available to download. Don't worry about making markdown cells, just make sure the code is there.