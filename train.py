from math import ceil
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import numpy as np
import time
import json

BATCH_SIZE=32
NUM_EPOCHS = 5

class TrainProcedure():
    
    def __init__(self):
        self.BATCH_SIZE=32
        self.NUM_EPOCHS = 5

    def trainHead(self, model, train_dataset, val_dataset, filename, batch_size=BATCH_SIZE, epoch=NUM_EPOCHS, lr=0.001):
        print('Learn the head')
        model.get_layer('efficientnetb0').trainable = False
        model.compile(optimizer= Adam(learning_rate=lr),
                    loss = SparseCategoricalCrossentropy(), run_eagerly=True,
                    metrics = ['accuracy'])

        pt = ceil(epoch/10)
        mc = ModelCheckpoint(filename+'.h5', monitor='val_loss', verbose=1, save_best_only=True)
        es=EarlyStopping(monitor='val_loss',patience=pt, restore_best_weights=True)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(buffer_size = AUTOTUNE)
        
        #val_dataset = val_dataset.batch(VAL_BATCH_SIZE)
        val_dataset = val_dataset.batch(len(val_dataset))
        val_dataset = val_dataset.prefetch(buffer_size = AUTOTUNE)

        history1 = model.fit(train_dataset, epochs=epoch, initial_epoch=0, callbacks=[mc,es], 
                            validation_data = val_dataset)
                            #steps_per_epoch=len(train_dataset)//BATCH_SIZE)
                            #validation_steps=len(val_dataset)//BATCH_SIZE)
        return history1

    #Fine tune the whole network
    def trainAll(self, model, train_dataset, val_dataset, filename, batch_size=BATCH_SIZE, epoch=NUM_EPOCHS, lr=0.0001):
       
        model.get_layer('efficientnetb0').trainable = True #unfreeze the base layers
        #model.load_weights(filename+'.h5')
        model.compile(optimizer= Adam(learning_rate=lr, decay=1e-6),
                    loss = SparseCategoricalCrossentropy(), run_eagerly=True,
                    metrics = ['accuracy'])
        
        pt = ceil(epoch/10)
        mc = ModelCheckpoint(filename + '_finetune.h5', monitor='val_loss', verbose=1, save_best_only=True)
        es=EarlyStopping(monitor='val_loss',patience=pt, restore_best_weights=True)
        
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(buffer_size = AUTOTUNE)
        
        #val_dataset = val_dataset.batch(BATCH_SIZE)
        val_dataset = val_dataset.batch(len(val_dataset))
        val_dataset = val_dataset.prefetch(buffer_size = AUTOTUNE)
        
        history2 = model.fit(train_dataset, epochs=epoch, initial_epoch=0, callbacks=[mc,es], 
                            validation_data = val_dataset) 
                            #validation_steps=len(val_dataset)//BATCH_SIZE)
        return history2
    
    def finetune_hyperparam(self, model,train_dataset, val_dataset, filename):
        lr = 0.0001
        BATCH_SIZE=32
        train_dataset = train_dataset.batch(BATCH_SIZE)
        train_dataset = train_dataset.prefetch(buffer_size = AUTOTUNE)
        
        val_dataset = val_dataset.batch(len(val_dataset))
        val_dataset = val_dataset.prefetch(buffer_size = AUTOTUNE)
        
        model.load_weights(filename + '_finetune.h5')
        model.compile(optimizer=SGD(learning_rate=lr), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])
        mc = ModelCheckpoint(filename+'_ft_sgd.h5', monitor='val_accuracy', verbose=1, save_best_only=True)
        es=EarlyStopping(monitor='val_accuracy',patience=0, restore_best_weights=True)
        THIRD_EPOCHS= 2
        
        history = model.fit(train_dataset, epochs=THIRD_EPOCHS, initial_epoch=0, callbacks=[mc,es], 
                            validation_data = val_dataset, 
                            steps_per_epoch=len(train_dataset)//BATCH_SIZE) 
                            #validation_steps=len(val_dataset)//BATCH_SIZE)
        return history
        
    def saveModel(self, model, filename):
        model.save_weights('weights.h5')
        model_json = model.to_json()
        with open('model.json', "w") as json_file:
            json_file.write(model_json)
            
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights.h5')
        loaded_model.save(filename)
    
    def saveModelHistory(self, d, name, histList):
        dt = time.time()
        for idx,h in enumerate(histList):
            filename = './history/{}/modelHist-{}-{}-{}.json'.format(d, name,idx, dt)
            with open(filename, 'w') as f:
                json.dump(h.history, f)
                
    def savePlotBestAccLoss(self, ftHistories,f):
        best_t_losses = [min(x.history['loss']) for x in ftHistories]
        best_t_accuracies = [max(x.history['accuracy']) for x in ftHistories]
        best_v_losses = [min(x.history['val_loss']) for x in ftHistories]
        best_v_accuracies = [max(x.history['val_accuracy']) for x in ftHistories]

        plot1 = plt.figure(num=1)
        plt.plot(best_t_accuracies, label = 'train')
        plt.plot(best_v_accuracies, label = 'validation')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('iteration')
        plt.legend()
        plt.savefig(fname=f+'-accuracy.png', bbox_inches='tight')
        plt.clf()

        plot2 = plt.figure(num=2)
        plt.plot(best_t_losses, label = 'train')
        plt.plot(best_v_losses, label = 'validation')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('iteration')
        plt.legend()
        plt.savefig(fname=f+'-loss.png', bbox_inches='tight')
        
    def savePlotIterations(self,df,fname):
        fig, ax = plt.subplots()
        for labels, dfi in df.groupby("Subject-LO"):
            dfi.plot(ax = ax, x = 'Iteration', y = 'Best Acc', label = labels)
        ax.legend(title = 'Best Acc per iterations',bbox_to_anchor=(0.5, 1.05), ncol=6)
        #fig.show()
        fig.savefig(fname+'-acc.png')
        
        fig, ax = plt.subplots()
        for labels, dfi in df.groupby("Subject-LO"):
            dfi.plot(ax = ax, x = 'Iteration', y = 'Best Loss', label = labels)
        ax.legend(title = 'Best Loss per iterations',bbox_to_anchor=(0.5, 1.05), ncol=6)
        #fig.show()
        fig.savefig(fname+'-loss.png')
        
        
    def savePlotEpochAccLoss(self, history1,f):
        plt.plot(history1.history['val_accuracy'], label = 'val_accuracy')
        plt.plot(history1.history['accuracy'], label = 'train_accuracy')
        plt.title('model head accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(fname=f+'-accuracy.png', bbox_inches='tight')

        plt.plot(history1.history['val_loss'], label = 'val_loss')
        plt.plot(history1.history['loss'], label = 'train_loss')
        plt.title('model head loss')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(fname=f+'-loss.png', bbox_inches='tight')
        
    def getTotalCounts(self, confMatrix, name):
        TruePositiveNo = len(np.diag(confMatrix))
        FalsePositive = []
        for i in range(3):
            FalsePositive.append(sum(confMatrix[:,i]) - confMatrix[i,i])
        FalsePositiveNo = len(FalsePositive)
        FalseNegative = []
        for i in range(3):
            FalseNegative.append(sum(confMatrix[i,:]) - confMatrix[i,i])
        FalseNegativeNo = len(FalseNegative)
        TrueNegative = []
        for i in range(3):
            temp = np.delete(confMatrix, i, 0)   # delete ith row
            temp = np.delete(confMatrix, i, 1)  # delete ith column
            TrueNegative.append(sum(sum(temp)))
        TrueNegativeNo = len(TrueNegative)

        with open(name, "w") as f:
            f.write('TP : {}\n'.format(TruePositiveNo))
            f.write('FP : {}\n'.format(FalsePositiveNo))
            f.write('FN : {}\n'.format(FalseNegativeNo))
            f.write('TN : {}\n'.format(TrueNegativeNo))

    def getCountsPerClass(self, cm):
        tp0 = cm[0,0]
        fn0 = cm[0,1] + cm[0,2]
        fp0 = cm[1,0] + cm[2,0]
        tp1 = cm[1,1]
        fn1 = cm[1,0] + cm [1,2]
        fp1 = cm[0,1] + cm[2,1]
        tp2 = cm[2,2]
        fn2 = cm[2,0] + cm[2,1]
        fp2 = cm[0,2] + cm[1,2]
        return [tp0,fn0,fp0,tp1,fn1,fp1,tp2,fn2,fp2]
    
    def calcFullScores(self,l, fn):
        N_0 = 250
        N_1 = 109
        N_2 = 83
        N_class = 3
        
        tp0,fn0,fp0,tp1,fn1,fp1,tp2,fn2,fp2 = l
        f1_0 = 0 if (tp0 == 0 and fn0 == 0 and fp0 == 0) else (2*tp0)/((2*tp0) + fp0 + fn0)
        f1_1 = 0 if (tp1 == 0 and fn1 == 0 and fp1 == 0) else (2*tp1)/((2*tp1) + fp1 + fn1)
        f1_2 = 0 if (tp2 == 0 and fn2 == 0 and fp2 == 0) else (2*tp2)/((2*tp2) + fp2 + fn2)
        uf1 = (f1_0 + f1_1 + f1_2)/N_class
        uar = ((tp0/N_0)+(tp1/N_1)+(tp2/N_2))/N_class
        
        with open(fn,'a') as f:
            f.write("TP0:{} FN0:{} FP0:{} TP1:{} FN1:{} FP1:{} TP2:{} FN2:{} FP2:{}".format(tp0,fn0,fp0,tp1,fn1,fp1,tp2,fn2,fp2))
            f.write("F1_negative: {}, F1_positive: {}, F1_surprise: {}\n".format(f1_0,f1_1,f1_2))
            f.write("UF1: {}, UAR: {}\n".format(uf1,uar))