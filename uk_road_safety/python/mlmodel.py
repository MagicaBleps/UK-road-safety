import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Lambda


def get_folds(
    df,
    fold_length,
    fold_stride):
    '''
    This function slides through the Time Series dataframe of shape (n_timesteps, n_features) to create folds
    - of equal `fold_length`
    - using `fold_stride` between each fold
    Returns a list of folds, each as a DataFrame
    '''
    folds=[]
    for index in range(0,df.shape[0]-fold_length,fold_stride):
        fold=df[index:index+fold_length]
        folds.append(fold)
    return folds


def train_test_split(fold,
                     train_test_ratio,
                     input_length):
    '''
    Returns a train dataframe and a test dataframe (fold_train, fold_test)
    from which one can sample (X,y) sequences.
    df_train should contain all the timesteps until round(train_test_ratio * len(fold))
    '''
    # TRAIN SET
    last_train_idx = round(train_test_ratio * len(fold))
    train_fold = fold.iloc[0:last_train_idx, :]


    # TEST SET
    first_test_idx = last_train_idx - input_length
    test_fold = fold.iloc[first_test_idx:, :]

    return (train_fold, test_fold)

def get_X_y_strides(fold, input_length, output_length, sequence_stride):
    '''
    - slides through a `fold` Time Series (2D array) to create sequences of equal
        * `input_length` for X,
        * `output_length` for y,
    using a temporal gap `sequence_stride` between each sequence
    - returns a list of sequences, each as a 2D-array time series
    '''

    X, y = [], []

    for i in range(0, len(fold), sequence_stride):
        # Exits the loop as soon as the last fold index would exceed the last index
        if (i + input_length + output_length) > len(fold):
            break
        X_i = fold.iloc[i:i + input_length, :]
        y_i = fold.iloc[i + input_length:i + input_length + output_length, :][['Accidents']]
        X.append(X_i)
        y.append(y_i)

    return np.array(X), np.array(y)

def plot_history(history):

    fig, ax = plt.subplots(1,2, figsize=(20,7))
    # --- LOSS: MSE ---
    ax[0].plot(history.history['loss'])
    ax[0].plot(history.history['val_loss'])
    ax[0].set_title('MSE')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='best')
    ax[0].grid(axis="x",linewidth=0.5)
    ax[0].grid(axis="y",linewidth=0.5)

    # --- METRICS:MAE ---

    ax[1].plot(history.history['mae'])
    ax[1].plot(history.history['val_mae'])
    ax[1].set_title('MAE')
    ax[1].set_ylabel('MAE')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='best')
    ax[1].grid(axis="x",linewidth=0.5)
    ax[1].grid(axis="y",linewidth=0.5)

    return ax

def init_model(X_train):

    # 0 - Normalization
    #normalizer = Normalization()
    #normalizer.adapt(X_train)

    #reg_l1_l2 = regularizers.l1_l2(l1=0.005, l2=0.005)
    # 1 - RNN architecture
    model = models.Sequential()
    ## 1.0 - All the rows will be standardized through the already adapted normalization layer
    #model.add(normalizer)
    ## 1.1 - Recurrent Layer
    model.add(layers.LSTM(30,
                          activation='tanh',
                          return_sequences = False,
                          recurrent_dropout = 0.3,
                          input_shape=X_train[0].shape))
    # model.add(layers.LSTM(20,
    #                     activation='tanh',
    #                     return_sequences = True,
    #                     recurrent_dropout = 0.3
    #                     ))
    ## 1.2 - Predictive Dense Layers
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dropout(rate=0.3))
    model.add(layers.Dense(1, activation='linear'))

    # 2 - Compiler
    # ======================
    adam = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model


def fit_model(model, X_train, y_train, verbose=1):

    es = EarlyStopping(monitor = "val_loss",
                      patience = 10,
                      mode = "min",
                      restore_best_weights = True)


    history = model.fit(X_train, y_train,
                        validation_split = 0.3,
                        shuffle = False,
                        batch_size = 16,
                        epochs = 500,
                        callbacks = [es],
                        verbose = verbose)

    return model, history

def init_baseline(output_length):

    model = models.Sequential()
    model.add(layers.Lambda(lambda x: x[:,-output_length:,0,None]))

    adam = optimizers.Adam(learning_rate=0.02)
    model.compile(loss='mse', optimizer=adam, metrics=["mae"])

    return model

def cross_validate_baseline_and_lstm(df, fold_length, fold_stride,
                                     train_test_ratio, input_length,
                                     output_length, sequence_stride):
    '''
    This function cross-validates
    - the "last seen value" baseline model
    - the RNN model
    '''

    list_of_mae_baseline_model = []
    list_of_mae_recurrent_model = []

    # 0 - Creating folds
    # =========================================
    folds = get_folds(df, fold_length, fold_stride)
    print(len(folds))

    for fold_id, fold in enumerate(folds):

        # 1 - Train/Test split the current fold + normalize
        # =========================================
        (fold_train, fold_test) = train_test_split(fold, train_test_ratio, input_length)
        columns=fold_train.columns
        scaler=MinMaxScaler()
        fold_train=pd.DataFrame(scaler.fit_transform(fold_train),columns=columns)
        fold_test=pd.DataFrame(scaler.transform(fold_test),columns=columns)

        X_train_scaled, y_train_scaled = get_X_y_strides(fold_train, input_length, output_length, sequence_stride)
        X_test_scaled, y_test_scaled = get_X_y_strides(fold_test, input_length, output_length, sequence_stride)

        y_train=y_train_scaled
        y_test=y_test_scaled

        for i,y in enumerate(y_train_scaled):
            y_train[i]=(scaler.inverse_transform(y))
        for i,y in enumerate(y_test_scaled):
            y_test[i]=(scaler.inverse_transform(y))

        y_test=y_test.astype(int)
        y_train=y_train.astype(int)

        # 2 - Modelling
        # =========================================

        ##### Baseline Model
        baseline_model = init_baseline(output_length)
        mae_baseline = baseline_model.evaluate(X_test_scaled, y_test, verbose=0)[1]
        list_of_mae_baseline_model.append(mae_baseline)
        print("-"*50)
        print(f"MAE baseline fold n°{fold_id} = {round(mae_baseline, 2)}")

        ##### LSTM Model
        model = init_model(X_train_scaled)
        es = EarlyStopping(monitor = "val_mae",
                           mode = "min",
                           patience = 2,
                           restore_best_weights = True)
        history = model.fit(X_train_scaled, y_train,
                            validation_split = 0.3,
                            shuffle = False,
                            batch_size = 32,
                            epochs = 50,
                            callbacks = [es],
                            verbose = 0)
        res = model.evaluate(X_test_scaled, y_test, verbose=0)
        mae_lstm = res[1]
        list_of_mae_recurrent_model.append(mae_lstm)
        print(f"MAE LSTM fold n°{fold_id} = {round(mae_lstm, 2)}")

        ##### Comparison LSTM vs Baseline for the current fold
        print(f"Improvement over baseline: {round((1 - (mae_lstm/mae_baseline))*100,2)} % \n")

    return list_of_mae_baseline_model, list_of_mae_recurrent_model

def plot_predictions(y_test, y_pred, y_bas):
    '''This function plots n_of_sequences plots displaying the original series and
    the two predictions (from the model and form the baseline model)'''
    plt.figure(figsize=(10, 5))
    for i,id in enumerate([0,10]):
        plt.subplot(1,2,i+1)
        df_test=pd.DataFrame(y_test[id])
        df_pred=pd.DataFrame(y_pred[id].astype(int))
        df_bas=pd.DataFrame(y_bas[id])
        plt.plot(df_test.values,c='black',label='test set')
        plt.plot(df_pred.values,c='orange',label='lstm prediction')
        plt.plot(df_bas.values,c='blue',label='baseline prediction')
    plt.show()
    return None
