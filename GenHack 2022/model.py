#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DICRECTORY* <!>
#
# See below an example of a generative model
# G_\theta(Z) = np.max(0, \theta.Z)
############################################################################

import numpy as np
#import os
#import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>
def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """
    # See below an example
    # ---------------------
    #latent_variable = noise[:, :10]  # use the first 10 dimensions of the noise
    latent_variable = noise[:,:50]
    
    df_train = pd.read_csv(os.getcwd()+'./data/df_train.csv')
    df = df_train.drop('dates',1)
    
    n_windows = 6330 # len(train_data) - seq_len = 9618 - 3288
    batch_size = 128
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df).astype(np.float32)
    
    # create 3288 * noise shape (3288,50)
    def make_random_data():
        while True:
            yield latent_variable

    random_series = iter(tf.data.Dataset
                         .from_generator(make_random_data, output_types=tf.float32)
                         .batch(batch_size)
                         .repeat())
    # model
    new_model = tf.keras.models.load_model('/parameters/saved_model/my_model')
    
    # generated data
    generated_data = []
    for i in range(int(n_windows / batch_size)):
        Z_ = next(random_series)
        d = new_model(Z_)
        generated_data.append(d)
        
    generated_data = np.array(np.vstack(generated_data))
    generated_data = (scaler.inverse_transform(generated_data.reshape(-1, n_seq)).reshape(-1, seq_len, n_seq))
        
    # load my parameters (of dimension 10 in this example). 
    # <!> be sure that they are stored in the parameters/ directory <!>
       # parameters = np.load(os.path.join("parameters", "param.npy"))
    
    return generated_data[0]




