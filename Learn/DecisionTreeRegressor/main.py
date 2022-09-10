from pandas import read_csv
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# import dataset to dataframe
data_path = "/home/ayoubamer/Workspace/ML/Kaggle/Datasets/melb_data.csv"
data = read_csv(data_path)
# summary
print(data.describe())
# print("Average Of Distance : ", data["Distance"].mean())


# Data Modeling
print("Columns : ", data.columns)
print()

# set input and output training data
chosenFeatures = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = data[chosenFeatures]  # input
y = data.Price  # output

# print("Input : ", x.head())
# print("Output : ", y)

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# building the model
# random state to get the same results ==> no affect on training
model = DecisionTreeRegressor()
model.fit(x_train, y_train)

# predict
pred = model.predict(x_test)
print(pred)

# accuracy
acc = model.score(x_test, y_test)
print("Accuracy : ", acc)

print("Algorithm Not Accurate.")
