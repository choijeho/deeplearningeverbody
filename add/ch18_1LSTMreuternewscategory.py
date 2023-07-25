from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, Activation
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import reuters
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test)=reuters.load_data(num_words=1000, test_split=0.2)
category=np.max(y_train)+1

X_train=sequence.pad_sequences(x_train, maxlen=100)
X_test=sequence.pad_sequences(x_test, maxlen=100)
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
model=Sequential()
model.add(Embedding(1000,10))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(46, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
estop=EarlyStopping(monitor='val_loss', patience=5)
history=model.fit(X_train, y_train, batch_size=20, epochs=200, validation_data=(X_test, y_test),
                  callbacks=[estop])
history=model.fit(X_train, y_train, batch_size=40, epochs=100, validation_split=0.25, callbacks=[estop])

print('\nTest Accuracy: %.4f' % (model.evaluate(X_test, y_test)[1]))

y_vloss=history.history['val_loss']
y_loss=history.history['loss']

x_len=np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
