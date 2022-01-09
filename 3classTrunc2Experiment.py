from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers.experimental.preprocessing import RandomContrast, RandomFlip
from tensorflow.keras.models import Model
from tensorflow.keras import Input, Sequential, regularizers
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from operator import add
from train import TrainProcedure
from losocv import LosoCv
from datetime import datetime
import os
import argparse
import util

try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--headIter', type=int, dest='noOfHeadIter', default=1)
    parser.add_argument('--ftIter', type=int, dest='noOfFtIter', default=5)
    parser.add_argument('--headEpoch', type=int, dest='noOfHeadEpoch', default=5)
    parser.add_argument('--ftEpoch', type=int, dest='noOfFtEpoch', default=10)
    parser.add_argument('--outputPath', dest='outp', default='output')
    parser.add_argument('--expName', dest='expName', default='head-base')
    args = parser.parse_args()
    print(args)

    fld = os.getcwd()
    datasetFile = fld + '\dataset-csv\combinedDataset3class.csv'
    weights = fld + '/trainingWeights'
    trainProc = TrainProcedure()
    loso = LosoCv(datasetFile)
    resultCounts = [0,0,0,0,0,0,0,0,0] #TP0, FN0, FP0, TP1, FN1, FP1, TP2, FN2, FP2

    INPUT_SHAPE = (224,224,3)
    effB0_model = EfficientNetB0(weights='imagenet', 
                    include_top=False, input_shape=INPUT_SHAPE)
    effB0_model.trainable = False
    #effB0_model.summary()
    truncEffB0Model = Model(inputs=effB0_model.input ,outputs=effB0_model.get_layer('block2b_add').output,  name='efficientnetb0')

    # ADD NEW TRAINABLE LAYERS ON TOP TO BUILD THE FINAL MODEL

    BASE_MODEL_PATH = '{}/3class-trunc2.h5'.format(weights)
    inputs = Input(shape=(224,224,3))
    #data_augmentation = Sequential([RandomFlip('horizontal'), RandomContrast(0.2)])
    #x = preprocess_input(inputs)
    x = truncEffB0Model(inputs, training=False) #run in inference mode
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3,activation='softmax', kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x) #3 output classes
    model = Model(inputs, outputs)
    model.save(BASE_MODEL_PATH)
    effB0_model.summary()

    today = datetime.now()
    dt = today.strftime("%d%m%Y%H")
    subjects = loso.getSubjects()
    dataH_train = []
    dataFt_train = []
    dataH_val = []
    dataFt_val = []
    outpFile = "{}/{}-{}.txt".format(args.outp,args.expName,dt)  

    for idx,subject in enumerate(subjects):
        print('Subject {} of 68'.format(idx+1))
        testSet = loso.getTestDataSet(subject)
        testSet = testSet.batch(len(testSet))
        trainDataset,valDataset = loso.getTrainValDataSet(subject)
        model = load_model(BASE_MODEL_PATH)

        #train
        
        for i in range(args.noOfHeadIter):
            print('-------Iteration Head: {}'.format(i)) 
            name = '{}/{}-head-{}-{}'.format(weights,subject,args.expName,dt)
            history1 = trainProc.trainHead(model,trainDataset, valDataset,name, epoch = args.noOfHeadEpoch, lr = 0.001)
            util.getBestAccLossTrain(history1,dataH_train,i,subject)
            util.getBestAccLossVal(history1,dataH_val,i,subject)

        for i in range(args.noOfFtIter):
            print('-------Iteration FT: {}'.format(i))
            name = '{}/{}-finetuning-{}-{}'.format(weights,subject,args.expName,dt)
            history1 = trainProc.trainAll(model,trainDataset, valDataset,name, epoch = args.noOfFtEpoch, lr = 0.00001)
            util.getBestAccLossTrain(history1,dataFt_train,i,subject)
            util.getBestAccLossVal(history1,dataFt_val,i,subject)
          
        #test
        print("Evaluating " + subject)
        classes = [0,1,2]
        res = model.predict(testSet)
        y_pred = [classes[np.argmax(p)] for p in res]
        #print(y_pred)
        y_actual = np.concatenate([y for x, y in testSet], axis=0)
        #print(y_actual)
        cm = confusion_matrix(y_actual,y_pred,labels=classes)
        #print(cm)
        cpc = trainProc.getCountsPerClass(cm)
        #print(cpc)
        resultCounts = list( map(add, resultCounts, cpc) )
        msg = 'Subject {}: Result counts: {}\n'.format(subject, cpc)
        util.sendNotification(args.expName + ' - ' + msg)
        with open(outpFile, 'a') as f:
            f.write(msg)
    
    util.saveListToCSV(dataH_train, '{}/{}-head-train-{}'.format(args.outp,args.expName,dt))
    util.saveListToCSV(dataH_val, '{}/{}-head-val-{}'.format(args.outp,args.expName,dt))
    util.saveListToCSV(dataFt_train, '{}/{}-ft-train-{}'.format(args.outp,args.expName,dt))
    util.saveListToCSV(dataFt_val, '{}/{}-ft-val-{}'.format(args.outp,args.expName,dt))
    
    dfH = pd.DataFrame(columns=['Iteration', 'Best Acc', 'Best Loss', 'Subject-LO'], data=dataH_val)
    trainProc.savePlotIterations(dfH,"{}/{}-Val-head-{}".format(args.outp,args.expName,dt))   
    
    dfFt = pd.DataFrame(columns=['Iteration', 'Best Acc', 'Best Loss', 'Subject-LO'], data=dataFt_val)
    trainProc.savePlotIterations(dfFt,"{}/{}-Val-ft-{}".format(args.outp,args.expName,dt))   

    trainProc.calcFullScores(resultCounts,outpFile)
    with open(outpFile, 'a') as f:
        f.write('\n {}'.format(args.__dict__))

except Exception as e:
    print(e)
    util.sendNotification('3 class Base Model Error')
else:
    util.sendNotification('3 class Base Model completes running')