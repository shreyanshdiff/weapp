import streamlit as st

   
    

st.title("Neural Network for MNIST dataset")

num_neurons = st.slider("Number of Neurons" , 1 , 64)
num_epochs = st.slider("Number of epochs" , 1 , 10)
activate = st.text_input('Activation')
"The number of neurons " + str(num_neurons)
"The activation function is " + activate
 
 
 
 

if st.button('Train the model'):
    import tensorflow as tf
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.layers import *
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import ModelCheckpoint

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    def preprocess_images(images):
        images = images / 255
        return images
    X_train = preprocess_images(X_train)
    X_test = preprocess_images(X_test)

    model = Sequential()
    model.add(InputLayer((28, 28)))
    model.add(Flatten())
    model.add(Dense(num_neurons, 'relu'))
    model.add(Dense(10))
    model.add(Softmax())
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    cp = ModelCheckpoint('model', save_best_only=True)
    history_cp=tf.keras.callbacks.CSVLogger('history.csv', separator=",", append=False)
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, callbacks=[cp, history_cp])
    

if st.button('Evaluate the model'):
    import pandas as pd
    import matplotlib.pyplot as plt
    history = pd.read_csv('history.csv')
    fig = plt.figure()
    plt.plot(history['epoch'], history['accuracy'], )
    plt.plot(history['epoch'], history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Val'], loc='lower right')
    fig

st.write("The numbers 0 through 9 are handwritten and comprise the MNIST collection. It contains 10,000 test photos that have been labelled or categorised in accordance with the training set of 60,000 images. An API is available to automatically download and extract images and labels in order to use the MNIST dataset with Keras.")