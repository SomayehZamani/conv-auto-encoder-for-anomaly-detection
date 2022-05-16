from sklearn.preprocessing import StandardScaler

import plot_util
import util
from dataset import Dataset

# comment this to use GPU
util.disableGPU()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, losses,Sequential
from tensorflow.keras.models import Model



houseNum=1
appliance_name='Dishwasher'
df:pd.DataFrame
dir = Dataset.getDataDir() + f'/{appliance_name}/{houseNum}/'
df=pd.read_pickle(dir+'/usage_events.pkl')

# make the length of signal data (feature dimension) dividable by 4
# because we use two consecutive convolution with strides equal to 2
size=df.shape[0]
size=size-size%4
df=df.iloc[:size]


featureDim=df.shape[0]
data = df.transpose().values

# split train and test data
trainSize=int(size*0.8)
train=data
test =train

# scaling train and test data with zero mean and unit variance of train data
scaler = StandardScaler()
scaler = scaler.fit(train)

train = scaler.transform(train)
test = scaler.transform(test)

# Defining Auto Encoder Model
# learning normal pattern of signal by encoding and reconstruct the data.
# the model learn to capture features of patterns in data in order to minimize reconstructing it.
# after training, signals that could not be reconstructed well will consider as anomaly.

# because the most information in data vector is in sequential nature of the signal
# and in order to ensure the model to learn features that are invariant to position of pattern
# in signal, we use convolutional layers.

class Conv1DAutoEncoder(Model):
  def __init__(self,input_size):
    super(Conv1DAutoEncoder, self).__init__()
    self.encoder = Sequential([
      # input shape
      layers.Input(shape=(input_size, 1)),
      # learn 32 kernel of size 7 (32 filters of size 7 with one bias)
      layers.Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
      # dropping 20% of output in training time to prevent over-fitting
      layers.Dropout(rate=0.2),
      # transform signal in lower dimensions
      layers.Conv1D(filters=8, kernel_size=7, padding="same", strides=2, activation="relu" )])

    # reversing the steps in encoder to reconstruct the original signal
    self.decoder = Sequential([
      layers.Conv1DTranspose(filters=8, kernel_size=7, padding="same", strides=2, activation="relu"),
      layers.Dropout(rate=0.2),
      layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
      layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
     # layers.Dense(input_size, activation="sigmoid")
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


autoencoder = Conv1DAutoEncoder(featureDim)

autoencoder.compile(optimizer='adam', loss='mae')

history = autoencoder.fit(train,train,
    epochs=100,batch_size=512,
    validation_split=0.1,shuffle=True,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")],
)

def plotModelLoss(history):
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

plotModelLoss(history)
encoded_data = autoencoder.encoder(test).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()


reconstructions = autoencoder.predict(train)
train_loss = tf.keras.losses.mae(np.squeeze(reconstructions), train)

# plotting the distribution of reconstruction error in training set
plt.hist(train_loss[None, :], bins=50)
plt.xlabel("Train loss")
plt.ylabel("No of examples")
plt.show()

# Choose a threshold for anomaly based on mean and standard deviation
# of reconstruction errors in training data

threshold = np.mean(train_loss) + np.std(train_loss)/2

reconstructions = autoencoder.predict(test)
test_loss = tf.keras.losses.mae(np.squeeze(reconstructions), test)

plt.hist(test_loss[None, :], bins=50)
plt.xlabel("Test loss")
plt.ylabel("No of examples")
plt.show()


# Classify data as an anomaly if the reconstruction error is greater than the threshold.
reconstructions = autoencoder(test)

errors = tf.keras.losses.mae(np.squeeze(reconstructions), test).numpy()
prediction=errors<threshold
# transform signal into original representation
inverseReconstruction=scaler.inverse_transform(np.squeeze(reconstructions))
rec=pd.DataFrame(inverseReconstruction.transpose())
rec.columns=df.columns # set the reconstruction name as same as signal names (that is date-time of starting)

# separate normal and anomaly samples in original dataset
normal=df.transpose()[prediction].transpose()
anomaly=df.transpose()[~prediction].transpose()

# separate normal and anomaly samples in reconstruction dataset
normalRec=rec.transpose()[prediction].transpose()
anomalyRec=rec.transpose()[~prediction].transpose()

# create list of anomaly errors
anomalyErrors=errors[~prediction]

# sort anomaly in order of reconstruction errors (from higher error to lower)
anomaly= anomaly.iloc[:,(-anomalyErrors).argsort()]
anomalyRec= anomalyRec.iloc[:,(-anomalyErrors).argsort()]

# plot histogram of total energy consumption
# for normal and anomaly signals

import seaborn as sns
title=f'total energy consumption histogram House #{houseNum}'
ax=normal.sum().hist(alpha=0.6)
anomaly.sum().hist(ax=ax,alpha=0.6)
plt.legend(['normal','anomaly'])
plt.xlabel('total energy consumption')
plt.ylabel('count')
plt.title(title)
plt.savefig(f'{dir}/{title}.png', dpi=300)
plt.show()


def plotSamplesError(real,rec,samples,title):
    # reconstruct list of samples using auto-encoder model and
    # transform each sample and reconstruction into original space
    # and plot each signal and reconstruction in one separate plot
    num=len(samples)
    rows = int(np.sqrt(num))
    columns = rows
    if rows * columns < num:
        columns += 1
    fig, axes = plot_util.getSubPlot(rows, columns, 5, 5)
    for i in range(num):
        r = int(i / columns)
        c = i % columns
        real.iloc[:,samples[i]].plot(c='b', ax=axes[r][c])
        rec.iloc[:,samples[i]].plot(c='r', ax=axes[r][c])
        axes[r][c].fill_between(np.arange(rec.iloc[:,samples[i]].shape[0]),
                                rec.iloc[:,samples[i]].values,
                                real.iloc[:,samples[i]].values,
                                color='lightcoral',
                                alpha=0.6)
        axes[r][c].set_title(f'date: {rec.columns[samples[i]]} - reconstruction error')
        axes[r][c].set_ylabel('Energy')
        axes[r][c].set_xlabel(f'time (in {10} sec)')

    plt.legend(labels=["Real", "Reconstruction", "Error"])
    fig.suptitle(title,fontsize=20)
    plt.savefig(f'{dir}/{title}.png', dpi=300)
    plt.show()


def plotSampleError(model,sample,scalar:StandardScaler=None):
    # reconstruct sample using auto-encoder model and
    # transform sample and reconstruction into original space
    # and plot both signal and reconstruction and the error between then
    sample=sample.reshape(1,-1)
    reconstruct=model(sample)
    sample=np.squeeze(sample)
    reconstruct=np.squeeze(reconstruct)
    if scalar is not None:
        sample = scalar.inverse_transform(sample)
        reconstruct=scaler.inverse_transform(reconstruct)
    plt.plot(sample, 'b')
    plt.plot(reconstruct, 'r')
    plt.fill_between(np.arange(sample.size), reconstruct, sample, color='lightcoral')
    plt.legend(labels=["Input", "Reconstruction", "Error"])
    plt.show()
plotSamplesError(normal,normalRec,list(range(9)),f'Samples of Normal dishwasher power usage for house #{houseNum}')
plotSamplesError(anomaly,anomalyRec,list(range(6)),f'Samples of anomaly in dishwasher power usage for house #{houseNum}')

