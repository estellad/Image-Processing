import numpy as np
import matplotlib.pyplot as plt
################################################################################################################
# Result
# if you have a list of value for y train, you can use encoder.transform(y_train) to encode y to binary model
# visualizing accuracy and loss

# Accuracy Plot
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
epochs = 100
xc=range(epochs)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

# Loss Plot
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
plt.figure(2,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'],loc=2)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


######################################## Optional Result ################################################
# Accuracy Numerical Value
from sklearn.metrics import accuracy_score
print('Predicting on test data')
# y_pred = np.rint(model.predict(X_val)) # For unaugmented
# print(accuracy_score(y_val, y_pred))
y_pred = np.rint(model.predict_generator(validation_generator, steps = 14))# For Augmented
# [[434 436]
#  [432 434]]
scores = model.evaluate_generator(validation_generator, 1) #1736 testing images
print("Accuracy = ", scores[1])


# Confusion Matrix
from sklearn.metrics import confusion_matrix
y_pred_unencoded = np.argmax(y_pred, axis=1)
y_val_unencoded = np.argmax(y_val, axis=1)
print(confusion_matrix(y_val_unencoded, y_pred_unencoded, labels=[0,1]))