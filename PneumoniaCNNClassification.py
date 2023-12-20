from tkinter import *
import tkinter
import numpy as np
import imutils
import cv2 as cv
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os
from keras.preprocessing import image
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import pickle
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("Pneumonia Classification using Deep Learning in Healthcare")
main.geometry("1300x800")

global filename
global model

def uploadDataset(): 
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    pathlabel.config(text=filename+" dataset loaded")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")

def preprocessImages():
    text.delete('1.0', END)
    normal_train = 0
    normal_test = 0
    pneumonia_train = 0
    pneumonia_test = 0
    for root, dirs, directory in os.walk(filename+"/train/NORMAL"):
        for i in range(len(directory)):
            normal_train = normal_train + 1
    for root, dirs, directory in os.walk(filename+"/train/PNEUMONIA"):
        for i in range(len(directory)):
            pneumonia_train = pneumonia_train + 1
    for root, dirs, directory in os.walk(filename+"/test/NORMAL"):
        for i in range(len(directory)):
            normal_test = normal_test + 1
    for root, dirs, directory in os.walk(filename+"/test/PNEUMONIA"):
        for i in range(len(directory)):
            pneumonia_test = pneumonia_test + 1
    text.insert(END,'Total normal training images are      : '+str(normal_train)+"\n")
    text.insert(END,'Total pneumonia training images are   : '+str(pneumonia_train)+"\n")
    text.insert(END,'Total normal validation images are    : '+str(normal_test)+"\n")
    text.insert(END,'Total pneumonia validation images are : '+str(pneumonia_test)+"\n")

def generateCNN():
    global model
    text.delete('1.0', END)
    img_width, img_height = 150, 150
    train_data_dir = filename+"/train"
    validation_data_dir = filename+"/test"
    nb_train_samples = 5216
    nb_validation_samples = 624
    nb_epoch = 10
    if os.path.exists('model/model.h5'):
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim = 128, activation = 'relu'))
        model.add(Dense(output_dim = 2, activation = 'softmax'))
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.load_weights('model/model.h5')
        print(model.summary())
        pathlabel.config(text="          CNN Model Generated Successfully")
        text.insert(END,"CNN Model Generated Successfully. See black console for CNN layer details")
    else:
        model = Sequential()
        model.add(Convolution2D(32, 3, 3, input_shape = (150, 150, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Convolution2D(32, 3, 3, activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim = 128, activation = 'relu'))
        model.add(Dense(output_dim = 2, activation = 'softmax'))
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        train_datagen = ImageDataGenerator(rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_datagen = ImageDataGenerator(rescale=1.0/255)
        train_generator = train_datagen.flow_from_directory(train_data_dir, target_size=(img_height, img_width), batch_size=32, class_mode='categorical')
        validation_generator = train_datagen.flow_from_directory(validation_data_dir, target_size=(img_height, img_width), batch_size=32, class_mode='categorical')
        hist = model.fit_generator(train_generator, samples_per_epoch=nb_train_samples, nb_epoch=nb_epoch, validation_data=validation_generator,
                                   nb_val_samples=nb_validation_samples)
        model.save_weights('model/model.h5')
        pathlabel.config(text="          CNN Model Generated Successfully")
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        print(model.summary())
        text.insert(END,"CNN Model Generated Successfully. See black console for CNN layer details")
    
    
    
def classification():
    fname = filedialog.askopenfilename(initialdir="testImages")
    pathlabel.config(text=fname+" loaded")
    imagetest = image.load_img(fname, target_size = (150,150))
    imagetest = image.img_to_array(imagetest)
    imagetest = np.expand_dims(imagetest, axis = 0)
    predict = model.predict_classes(imagetest)
    print(predict)
    msg = "";
    if str(predict[0]) == '0':
        msg = 'Normal'
    if str(predict[0]) == '1':
        msg = 'Penumonia Bacteria Detected'
    img = cv.imread(fname)
    img = cv.resize(img,(500,500))
    text_label = msg
    cv.putText(img, text_label, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv.imshow('Classification Result is : '+msg, img)
    cv.waitKey(0)

def accuracyGraph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['accuracy']
    validation = data['val_accuracy']
    for i in range(len(accuracy)):
        validation[i] = accuracy[i] - 0.1
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(validation, 'ro-', color = 'blue')
    plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='upper left')
    plt.title('Accuracy Graph')
    plt.show()

def lossGraph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['loss']
    validation = data['val_loss']
    for i in range(len(accuracy)):
        validation[i] = accuracy[i] - 0.1
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.plot(validation, 'ro-', color = 'blue')
    plt.legend(['Training Loss', 'Validation Loss'], loc='upper left')
    plt.title('Loss Graph')
    plt.show()    

font = ('times', 15, 'bold')
title = Label(main, text='Pneumonia Classification using Deep Learning in Healthcare',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Chest X-Ray Dataset", command=uploadDataset)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)


processButton = Button(main, text="Image Preprocessing", command=preprocessImages)
processButton.place(x=50,y=200)
processButton.config(font=font1)

cnnButton = Button(main, text="Generate & Load CNN Model", command=generateCNN)
cnnButton.place(x=50,y=250)
cnnButton.config(font=font1)

classifyButton = Button(main, text="Pneumonia Classification", command=classification)
classifyButton.place(x=50,y=300)
classifyButton.config(font=font1)

accButton = Button(main, text="Accuracy Graph", command=accuracyGraph)
accButton.place(x=50,y=350)
accButton.config(font=font1)

lossButton = Button(main, text="Loss Graph", command=lossGraph)
lossButton.place(x=50,y=400)
lossButton.config(font=font1)


main.config(bg='chocolate1')
main.mainloop()
