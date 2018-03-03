import keras, time
from keras import Sequential
from keras.layers import Dropout, Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
from keras.models import load_model
'''
#####################################################
--------------- MODEL STRUCTURE ---------------------
#####################################################
#For this project, basic model structures will be created:
#Concatenation of denseLayers with the same number of neurones each one.
#Dropout(True False) after dense Layers and DropoutRate selection

'''
def modelStructure(X_train, denseLayers, denseNeurons, dropout, dropoutRate, n_classes):
    keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(denseNeurons,
                    input_dim = X_train.shape[1], 
                    activation='relu', 
                    use_bias=True))
    if dropout == True: 
        model.add(Dropout(dropoutRate))
    for i in range(denseLayers - 1):
        model.add(Dense(denseNeurons, activation='relu'))
        if dropout == True: 
            model.add(Dropout(dropoutRate))
    model.add(Dense(n_classes, activation='softmax'))
    print(model.summary())
    return model

'''
#####################################################
--------------- MODEL CALLBACKS ---------------------
#####################################################
'''
#Log of accuracy and loss
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        
    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
'''        
#####################################################
----------------- MODEL COMPILE ---------------------
#####################################################
'''
# Two optimizers available (adam, rmsprop)
def modelCompile(optimizer, model):   
    if optimizer == 'adam':
        optCompile = keras.optimizers.Adam(lr=0.001)
    else: optCompile = 'rmsprop'
    
    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer= optCompile,
                  metrics=['accuracy'])
    print('Model Compiled')
    return model
'''
#####################################################
------------------- MODEL RUN -----------------------
#####################################################
'''
def modelHistory(model, batch_size, epochs, X_train, y_train_cat, X_test, y_test_cat,
                 bestModelPathLoss, bestModelPathAcc):
    history = AccuracyHistory()
    bestModelAcc = ModelCheckpoint(bestModelPathAcc, monitor="val_acc",
                      save_best_only=True, save_weights_only=False)
    bestModelLoss = ModelCheckpoint(bestModelPathLoss, monitor="val_loss",
                      save_best_only=True, save_weights_only=False)
    return model.fit(X_train, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test_cat),
            callbacks=[history, bestModelAcc, bestModelLoss]), history

'''
#####################################################
----------------- MODEL EVALUATION ------------------
#####################################################
'''    
def plotAccuracyLoss(path, name, history):
#PLOT ACCURACY AND LOSS EVOLUTION FOR TRAINING AND TESTING SETS
    plt.plot(range(len(history.acc)), history.acc)
    plt.plot(range(len(history.acc)), history.val_acc)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(path + name + '_acc.png')
    plt.show()
    plt.plot(range(len(history.loss)), history.loss)
    plt.plot(range(len(history.loss)), history.val_loss)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(path + name + '_loss.png')
    plt.show()

'''
#####################################################
---------------- NEURAL NET PROCESS -----------------
#####################################################
'''  
def NeuralNetProcess(baseName, optimizer, batch_size, epochs, denseLayers, layerNeurons, dropout, dropoutRate,
                     X_train, X_test, y_train_cat, y_test_cat, n_classes):
    name = baseName + optimizer + '_b' + str(batch_size) + '_e' + str(epochs) + '_DL' +  \
            str(denseLayers) + '_' + str(layerNeurons) + '_drop-' + str(dropout) + '_' + str(dropoutRate)
    print('--------------- MODEL STRUCTURE ---------------------')
    model = modelStructure(X_train, denseLayers, layerNeurons, dropout, dropoutRate, n_classes)
    print('----------------- MODEL COMPILE ---------------------')
    modelCompile(optimizer, model)
    bestModelPath = os.getcwd() + '/NNbestModel/'
    bestModelPathAcc = bestModelPath + 'model_acc_' + name + '.hdf5'
    bestModelPathLoss = bestModelPath + 'model_loss_' + name + '.hdf5'
    print('------------------- MODEL RUN -----------------------')
    start = time.time()
    modhistory, history = modelHistory(model, batch_size, epochs, X_train, y_train_cat, X_test, y_test_cat,
                     bestModelPathLoss, bestModelPathAcc)
    end = time.time()
    train_time = end-start
    print('Elapsed time: ' + str(round(train_time,2)) + ' seconds')
    print('----------------- MODEL EVALUATION ------------------')
    plotAccuracyLoss(os.getcwd() + '/NNpics/', name, history)
    print('-------------------- MODEL LOAD  --------------------')
    #best_model_acc = load_model(bestModelPathAcc)
    best_model_loss = load_model(bestModelPathLoss)
    print('----------------- MODEL PREDICTION ------------------')
    y_pred_mlp = best_model_loss.predict(X_test, verbose = 0)
    return name, y_pred_mlp, train_time

def NeuralNetPredict(path, X):
    model = load_model(path)
    start = time.time()
    y_pred = model.predict(X, verbose = 0)
    end = time.time()
    process_time = end-start
    return y_pred, process_time
