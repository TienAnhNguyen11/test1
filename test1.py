import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

iris_data = load_iris()

x = iris_data.data
y_ = iris_data.target.reshape(-1, 1)

encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)


train_val_x, test_x, train_val_y, test_y = train_test_split(x, y, test_size=0.2)
train_x, val_x, train_y, val_y = train_test_split(train_val_x, train_val_y, test_size=0.2)

model = Sequential()

model.add(Dense(10, input_shape=(4,), activation="relu", name="fc1"))
model.add(Dense(10, activation="relu", name="fc2"))
model.add(Dense(3, activation="softmax", name="output"))

optimizer = Adam(lr=0.001)
model.compile(optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(train_x, train_y, validation_data=(val_x, val_y), batch_size=8, epochs=200)

results = model.evaluate(test_x, test_y)

print("loss: ", results[0])
print("accuracy: ", results[1])

y_predict = np.argmax(model.predict(test_x), axis=1)

y_fact = np.argmax(test_y, axis=1)

print(y_predict)
print(y_fact)




