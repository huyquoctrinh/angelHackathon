from sklearn.model_selection import train_test_split
from glob import glob
from shutil import copyfile

list_images_1 = glob('hen1/*.jpg')
list_labels_1 = glob('hen1/*.txt')

list_images_2 = glob('hen2/*.jpg')
list_labels_2 = glob('hen2/*.txt')

list_images_3 = glob('hen3/*.jpg')
list_labels_3 = glob('hen3/*.txt')

list_image_total = list_images_1 + list_images_2 + list_images_3
list_labels_total = list_labels_1 + list_labels_2 + list_labels_3

X_train, X_test, y_train, y_test = train_test_split(list_image_total, list_labels_total, test_size=0.2, random_state=42)

for i in range(len(X_train)):
    copyfile(X_train[i], 'final/train/' + "images/" + f"{i}.jpg")
    copyfile(y_train[i], 'final/train/' + "labels/" + f"{i}.txt")

for i in range(len(X_test)):
    copyfile(X_test[i], 'final/val/' + "images/" + f"{i}.jpg")
    copyfile(y_test[i], 'final/val/' + "labels/" + f"{i}.txt")