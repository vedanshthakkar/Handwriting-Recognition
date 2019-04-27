from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from hog_code import HOG
from dataset_code import dataset
import argparse

#Now, instead of using argparse, just load the dataset "train.csv" directly in the argument of dataset.load_digits. See line 14
# And define the path directly into the argument of joblib.dump. This is the path to where you want to store the model. See line 27'''


#ap = argparse.ArgumentParser()
#ap.add_argument("-d","--dataset", required = True, help = r'C:\Users\vedan\PycharmProjects\classify\train.csv')
#ap.add_argument("-m","--model", required = True, help = r'C:\Users\vedan\PycharmProjects\classify\"svm.cPickle')
#args = vars(ap.parse_args())

(digits, target) = dataset.load_digits('train.csv')
data = []
hog = HOG(orientations = 18, pixelsPerCell = (10, 10),cellsPerBlock = (1, 1), transform = True)

for image in digits:
    image = dataset.deskew(image, 20)
    image = dataset.center_extent(image, (20, 20))

    hist = hog.describe(image)
    data.append(hist)

model = LinearSVC(random_state = 42)
model.fit(data, target)
joblib.dump(model, r'C:\Users\vedan\PycharmProjects\classify\svm.cPickle')