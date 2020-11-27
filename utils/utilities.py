# TODO comment and refactor

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_log_error
import numpy as np

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def plot_history(history):
    err = history.history['mean_squared_error']
    val_err = history.history['val_mean_squared_error']
    err2 = history.history['mean_squared_logarithmic_error']
    val_err2 = history.history['val_mean_squared_logarithmic_error']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(err) + 1)

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 3, 1)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('loss')
    #plt.ylim(-0.5, 3)
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(x, err, 'b', label='Training mean_squared_error')
    plt.plot(x, val_err, 'r', label='Validation mean_squared_error')
    plt.title('mean_squared_error')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(x, err2, 'b', label='Training mean_squared_logarithmic_error')
    plt.plot(x, val_err2, 'r', label='Validation mean_squared_logarithmic_error')
    plt.title('mean_squared_logarithmic_error')
    plt.legend()

    plt.show()
    