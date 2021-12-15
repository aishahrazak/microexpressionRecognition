from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
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

        mc = ModelCheckpoint(filename+'.h5', monitor='val_loss', verbose=1, save_best_only=True)
        es=EarlyStopping(monitor='val_loss',patience=2, restore_best_weights=True)
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
        print('Fine-tuning whole network')
        
        model.get_layer('efficientnetb0').trainable = True #unfreeze the base layers
        #model.load_weights(filename+'.h5')
        model.compile(optimizer= Adam(learning_rate=lr, decay=1e-6),
                    loss = SparseCategoricalCrossentropy(), run_eagerly=True,
                    metrics = ['accuracy'])
        mc = ModelCheckpoint(filename + '_finetune.h5', monitor='val_loss', verbose=1, save_best_only=True)
        es=EarlyStopping(monitor='val_loss',patience=2, restore_best_weights=True)
        
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
                
    def plotBestAccLoss(self, ftHistories):
        best_t_losses = [min(x.history['loss']) for x in ftHistories]
        best_t_accuracies = [max(x.history['accuracy']) for x in ftHistories]
        best_v_losses = [min(x.history['val_loss']) for x in ftHistories]
        best_v_accuracies = [max(x.history['val_accuracy']) for x in ftHistories]

        plt.plot(best_t_accuracies, label = 'train')
        plt.plot(best_v_accuracies, label = 'validation')
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('iteration')
        plt.legend()
        plt.show()

        plt.plot(best_t_losses, label = 'train')
        plt.plot(best_v_losses, label = 'validation')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('iteration')
        plt.legend()
        plt.show()
        
    def plotEpochAccLoss(self, history1):
        plt.plot(history1.history['val_accuracy'], label = 'val_accuracy')
        plt.plot(history1.history['accuracy'], label = 'train_accuracy')
        plt.title('model head accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()

        plt.plot(history1.history['val_loss'], label = 'val_loss')
        plt.plot(history1.history['loss'], label = 'train_loss')
        plt.title('model head loss')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()