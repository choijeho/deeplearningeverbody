from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, Activation
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.callbacks import EarlyStopping
from attention import Attention

import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test)=imdb.load_data(num_words=5000)
X_train=sequence.pad_sequences(x_train, maxlen=500)
X_test=sequence.pad_sequences(x_test, maxlen=500)

model=Sequential()
model.add(Embedding(5000,100))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=True))
model.add(Attention())
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
estop=EarlyStopping(monitor='val_loss', patience=3)
history=model.fit(X_train, y_train, batch_size=40, epochs=100,
                  validation_data=(X_test, y_test), callbacks=[estop])

print('\nTest Accuracy: %.4f' % (model.evaluate(X_test, y_test)[1]))

y_vloss=history.history['val_loss']
y_loss=history.history['loss']

x_len=np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')
plt.show()

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
