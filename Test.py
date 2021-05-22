from keras import backend as K
K.set_image_dim_ordering('th')
import os, cv2
import numpy as np
from scipy.misc import toimage, imsave
import glob,shutil
from RealModel import model
from sklearn.metrics import accuracy_score

def show_img(image):
    toimage(image).show()
#################################################################################################################
# Test Data

img_rows = 100
img_cols = 100
num_channel = 1

num_classes = 2
names = ['cancer', 'normal']
# num_epoch = 20

path4 = 'C:\\Users\\edong\\PycharmProjects\\proj.py\\Test' # where test img and mask saved
newpath = 'C:\\Users\\edong\\PycharmProjects\\proj.py\\Test_Masked'

# Test Data Processing
# Data with Mask, change color, resize,label, split
def get_test_data (path, dst_dir):
    X = []
    datsetlist = os.listdir(path)
    total = len(datsetlist)
    for dataset in datsetlist:
        os.chdir(path + '/' + dataset)
        testtiffs = glob.glob('*.tif')
        it = iter(testtiffs)
        for img in it:
            shutil.copy(img, dst_dir)
            mask = next(it)
            img_name = os.path.split(img)[-1]
            masked_img = cv2.bitwise_and(img, img, mask=mask)
            gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
            resize_masked = cv2.resize(gray_masked, (img_rows, img_cols))
            imsave(os.path.join(dst_dir + '\\' + dataset, img_name), resize_masked)
            X.append(resize_masked)
    X = np.asarray(X).astype('float32')
    X = X * 1. / 255.
    return X

X = get_test_data(path4,newpath)
# Test Data Labeling


# Test Data Prediction
# TODO: figure out model.predict_proba
maskedsetlist = os.listdir(newpath)
total = len(maskedsetlist)
# accuracy_rate = []
ind = 0
for dataset in maskedsetlist:
    y_pred = np.rint(model.predict_proba(dataset))
    # accuracy_rate[ind]= accuracy_score(y_test, y_pred)
    ind += 1
#################################################################################################################
# write out the csv file
first_row = 'file_name, num_cells, cancer_probability, std_cancer_probability, accuracy_rate, '
doc_name = 'result.csv'

# TODO: sklearn.metrics.model.predict_proba
#LDO_Score: avg probabilities
cancer_probability = []
std_list= []
nlist = []
with open(doc_name, 'w+') as f:
    f.write(first_row + '\n')
    for i in range(total):
        s = maskedsetlist[i] + ',' + nlist[i] + ','+ cancer_probability[i]+ ',' + std_list[i] #+ ',' + accuracy_rate[i]
        f.write(s + '\n')
