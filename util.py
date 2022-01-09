import telegram_send
import csv

def sendNotification(msg):
    #token = '5014749052:AAEq67L0O6s-gQxhc8rBzFydmrSoyZNPnzc'
    telegram_send.send(messages=[msg])
    
def getBestAccLossTrain(history,data,i,subject):
    best_t_accuracy = max(history.history['accuracy'])
    best_t_loss = min(history.history['loss'])
    
    data.append([i,best_t_accuracy,best_t_loss,subject])
        
def getBestAccLossVal(history,data,i,subject):
    best_v_accuracy = max(history.history['val_accuracy'])
    best_v_loss = min(history.history['val_loss'])
    
    data.append([i,best_v_accuracy, best_v_loss,subject])
    
def saveListToCSV(l, name):
    with open(name + '.csv', "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(l)
    
#sendNotification('test message')