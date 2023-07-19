from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, Activation, \
    LeakyReLU, BatchNormalization, Reshape, UpSampling2D, Conv2D, Flatten, Input
from tensorflow.keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.datasets import imdb, mnist
from keras.callbacks import EarlyStopping
from attention import Attention

import numpy as np
import matplotlib.pyplot as plt

generator=Sequential()
generator.add(Dense(128*7*7, input_dim=100, activation='LeakyReLU'))
# 128 임의, 7x7 원래 28x28-> 7을 2배->14를 두배
generator.add(BatchNormalization())
generator.add(Reshape((7,7,128)))
generator.add(UpSampling2D())
generator.add(Conv2D(64, kernel_size=5, padding='same'))
generator.add(BatchNormalization())
generator.add(Activation(LeakyReLU(0.2)))
generator.add(UpSampling2D())
generator.add(Conv2D(1, kernel_size=5, padding='same', activation='tanh'))

discriminator=Sequential()
discriminator.add(Conv2D(64, kernel_size=5, strides=2, input_shape=(28,28,1), padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=5, strides=2, padding='same'))
discriminator.add(Activation(LeakyReLU(0.2)))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer='adam')
discriminator.trainable=False

ginput=Input(shape=(100,))
dis_output=discriminator(generator(ginput))
gan=Model(ginput, dis_output)
gan.compile(loss='binary_crossentropy', optimizer='adam')

def gan_train(epoch, batch_size, saving_interval):
    (X_train, _), (_, _) = mnist.load_data()
    X_train=X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_train=(X_train-127.5)/127.5
    true=np.ones((batch_size, 1))
    fake=np.zeros((batch_size, 1))

    for i in range(epoch):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs =X_train[idx]
        d_loss_real = discriminator.train_on_batch(imgs, true)
        noise = np.random.normal(0,1,(batch_size,100))
        gen_imgs = generator.predict(noise)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        g_loss = gan.train_on_batch(noise, true)
        print('epoch: %d' % i, ' d_loss: %.4f' % d_loss, ' g_loss:%.4f' % g_loss)
        if i % saving_interval == 0:
            noise=np.random.normal(0,1,(25,100))
            gen_imgs = generator.predict(noise)
            gen_imgs= 0.5 * gen_imgs + 0.5
            fig, axs=plt.subplots(5,5)
            count=0
            for j in range(5):
                for k in range(5):
                    axs[j,k].imshow(gen_imgs[count, :, :, 0], cmap='gray')
                    axs[j,k].axis('off')
                    count += 1
            fig.savefig("gan_images/gan_mnist_%d.png" % i)
gan_train(2001,32,200)
