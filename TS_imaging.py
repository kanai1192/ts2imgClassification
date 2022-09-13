import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Activation, MaxPooling2D, Input, Conv1D, GlobalAveragePooling1D, GlobalAveragePooling2D,  BatchNormalization, Layer, Add, MaxPool2D
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from pyts.image import RecurrencePlot,GramianAngularField

#Defined VGG11 architecture
def VGG11(X_train, num_classes, lr = 0.0001):
    model = keras.models.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding="same", activation='relu', input_shape=(X_train.shape[1: ])),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    #keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
    ])

    adam = optimizers.Adam(lr = lr)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model

#Defined alexnet architecture
def alexnet(X_train, num_classes, lr = 0.0001):
    model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(X_train.shape[1: ])),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
    ])

    adam = optimizers.Adam(lr = lr)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])

    return model


def basic_cnn_functional(X_train, num_classes, lr = 0.0001):
    
    input1 = Input(shape = X_train.shape[1: ])
    cnn = Conv2D(filters = 128, kernel_size = (3,3), strides = (1,1), activation='relu', padding = 'same')(input1)
    cnn1 = Conv2D(filters = 64, kernel_size = (7,7), strides = (2,2), activation='relu', padding = 'same')(cnn)
    cnn2 = Conv2D(filters = 32, kernel_size = (7,7), strides = (1,1), activation='relu', padding = 'same')(cnn1)
    cnn3 = Conv2D(filters = 16, kernel_size = (7,7), strides = (1,1), padding = 'same')(cnn2)
    act = Activation('relu')(cnn3)
    maxP = MaxPooling2D(pool_size = (2,2))(act)

    # prior layer should be flattend to be connected to dense layers
    Flatt = Flatten()(maxP)
    # dense layer with 64 neurons
    dense = Dense(64, activation = 'relu')(Flatt)
    # final layer with neurons to classify the instances
    output = Dense(num_classes, activation = 'softmax')(dense)
    
    adam = optimizers.Adam(lr = lr)
    model = keras.models.Model(inputs=input1, outputs=output)
    model.compile(loss = 'categorical_crossentropy', optimizer = adam, metrics = ['accuracy'])
    
    return model

def CNN_GAF():

    #load raw anomaly TS data
    df = pd.read_pickle("dataset_TS_bin_class.pkl")   
    X = df.loc[:,df.columns[:-2]].to_numpy()

    #load binary labels "y" and multiclass labels Y_anom
    y = df['anomaly'].to_numpy()
    Y_anom = df["type"].to_numpy(dtype=np.uint8)
    num_classes = np.unique(Y_anom).shape[0]
    
    #----------Uncomment this block if you want to test recurrence plot transfomration and comment the GAF block both use pyts library (https://pyts.readthedocs.io/en/stable/index.html)----------------------
    #RP = RecurrencePlot()
    # X_rp = RP.fit_transform(X)
    # X_rp = np.float32(X_rp)
    # X_rp = X_rp.reshape(X_rp.shape[0], X_rp.shape[1], X_rp.shape[2], 1)
    #---------------------------------------------------------------------
    #--------------------------------GAF BLOCK-------------------------------------
    gasf = GramianAngularField(image_size=300, method='difference') #'difference' 'summation' (choose depending if you want gramian summation or difference field)
    X_gasf = gasf.fit_transform(X)
    X_gasf = np.float32(X_gasf)
    X_gasf = X_gasf.reshape(X_gasf.shape[0], X_gasf.shape[1], X_gasf.shape[2], 1)
    #---------------------------------------------------------------------
    
    y_bin = tf.keras.utils.to_categorical(y, num_classes=2) 
    y_multi = tf.keras.utils.to_categorical(Y_anom, num_classes=5)

    #---------------------Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_gasf, y_multi, test_size=0.33) #change X_gasf for X_rp if you want to test RP transformation


    
    #____________________MODEL SELECTION AND TRAINING___________________________
    
    model = basic_cnn_functional(X_train, num_classes=5, lr = 0.0001) #set number of classes based on the problem and set learning rate if needed (binary, multiclass)
    
    #There are more samples for class 0 than other, so we asign weights to compensate for that
    class_weight = {0: 0.1,
                    1: 1.,
                    2: 1.,
                    3: 1.,
                    4: 1.}

    history = model.fit(X_train, y_train, batch_size = 16, validation_split = 0.2, epochs = 10, verbose = 2, class_weight=class_weight)

    #----------------model evaluation-----------------------------------------
    results = model.evaluate(X_test, y_test, verbose = 2)
    print('Test accuracy: ', results[1])
    Y_pred = model.predict(X_test, verbose = 2)
    y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(y_test, axis=-1)
    print(confusion_matrix(Y_test, y_pred))
    target_names = ["noanomaly", "norecovery","recovery", "spikes", "slow"]
    print(classification_report(Y_test, y_pred, target_names=target_names))
    



if __name__ == "__main__":
    CNN_GAF()