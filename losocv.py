import csv
import numpy as np
import tensorflow as tf
import random

from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

class LosoCv():
    
    def __init__(self, csvFile=None):
        if(csvFile is None):
            csvFile = './dataset-csv/combinedDataset.csv'
        
        with open(csvFile) as f:
            reader = csv.reader(f, delimiter=',')
            self.data = list(reader) 
        samplesNum = len(self.data)
        self.trainNum = int(np.floor(0.8 * samplesNum)) #80-20 split

    def getSubjects(self):
        subjects = [x[0].removesuffix('.jpg').rsplit('/',1)[1].split('_',1)[0] for x in self.data]
        subjects = list(set(subjects))
        random.shuffle(subjects)
        return subjects

    def _im_file_to_tensor(self, file, label):
        im = tf.image.decode_jpeg(tf.io.read_file(file), channels=3)
        im = tf.image.resize(im,[224,224])
        label = tf.strings.to_number(label)
        #print(im.shape)
        return im, label

    def getTestDataSet(self, holdOutSubject):
        dataset = [x for x in self.data if holdOutSubject in x[0]]
        random.shuffle(dataset)
            
        filenames = tf.constant([x[0] for x in dataset])
        labels = tf.constant([x[1] for x in dataset])

        testDataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
        testDataset = testDataset.map(self._im_file_to_tensor,num_parallel_calls=AUTOTUNE)
        
        return testDataset
        
    def getTrainValDataSet(self, holdOutSubject):
        dataset = [x for x in self.data if holdOutSubject not in x[0]]
        random.shuffle(dataset)
        train_set = dataset[0:self.trainNum]
        val_set = dataset[self.trainNum:]
        
        train = tf.constant([x[0] for x in train_set])
        trainLabels = tf.constant([x[1] for x in train_set])
        val = tf.constant([x[0] for x in val_set])
        valLabels = tf.constant([x[1] for x in val_set])
        
        trainDataset = tf.data.Dataset.from_tensor_slices((train, trainLabels))
        trainDataset = trainDataset.map(self._im_file_to_tensor, num_parallel_calls=AUTOTUNE)
        valDataset = tf.data.Dataset.from_tensor_slices((val, valLabels))
        valDataset = valDataset.map(self._im_file_to_tensor,num_parallel_calls=AUTOTUNE)
    
        return trainDataset, valDataset

""" loso = LosoCv()
subjects = loso.getSubjects()
subject = subjects[0]
ds = loso.getTestDataSet(subject)
ds = ds.enumerate()
for e in ds.as_numpy_iterator():
    print(e) """