
#Importing the Modules and Classes
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.layers import Activation,Dropout,Flatten,Conv2D,MaxPooling2D,Dense
import time

#Images dir
#Please Locate the directory for test and train images 'FYI NO SPACES for Folder name'
Train_dir = 'D:/images/Model_and_files/downloads/train' #Train Directory location
Test_dir = 'D:/images/Model_and_files/downloads/test'   #Test Directory location

#Declaring Variables for testing different models
dense_layers =[0,1,2]  		#Providing the list of for choosing number of dense layer
Nodes_list = [32,64,128] 	#Providing the list number of nodes 
conv_layers = [1,2,3] 		#Providing the list for number of conv layers




#Creating the class of Image Data Generator with random transformations on the original image.
image_gen = ImageDataGenerator(rotation_range=30,
                              width_shift_range = 0.1,
                              height_shift_range = 0.1,
                              rescale = 1/255,
                              shear_range= 0.2,
                              zoom_range=0.2,
                              horizontal_flip = True,
                              fill_mode = 'nearest')



#Assinging the directory to Image_gen for Train and Test images
batch_size =32 				#Taking a batch of 32 images
input_shape=(150,150,3) 	#Setting the size of input images, since every image have different size

#Creating the random images for train dataset by Data augmentation process
#Target size is set to take only 2 dimensions 
#Class_mode is Categorical due to Multi-Class Classification 
train_image_gen = image_gen.flow_from_directory(Train_dir,
                                               target_size = input_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='categorical') 



#Creating the random images for test dataset
test_image_gen = image_gen.flow_from_directory(Test_dir,
                                               target_size = input_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='categorical')


#Using for loop for testing different number of Node size, Dense layer & Conv2D 
for dense_layer in dense_layers:
    for Num_Nodes in Nodes_list:
        for conv_layer in conv_layers:
            
            #Naming the model using Number of Conv2D, nodes and dense layer used
            Name = "{}-conv-{}-nodes{}-dense-{}".format(conv_layer, Num_Nodes, dense_layer, int(time.time()))

            #Initizaling the Sequential Class
            model = Sequential()

            #Adding the Conv2D layer with 'Relu' activation layer 
            model.add(Conv2D(filters=Num_Nodes,kernel_size=(3,3),input_shape=(150,150,3),activation='relu')) 	#first layer
            #Adding the MaxPooling2D layer with a pool size of (2,2)
            model.add(MaxPooling2D(pool_size=(2,2))) 

            #for loop for testing different number of Hidden Conv2D layers
            for l in range(conv_layer-1):
                model.add(Conv2D(filters=Num_Nodes,kernel_size=(3,3),activation='relu')) #Adding Conv2D layer with 'Relu' activation function
                model.add(MaxPooling2D(pool_size=(2,2))) #Performing Max Pooling on the image

            #Adding the Flatten layer to convert it into 1D 
            model.add(Flatten())

            #For loop for testing different number of Dense Layers
            for l in range(dense_layer):
                model.add(Dense(Num_Nodes)) 	# Adding the dense layer
                model.add(Activation('relu')) 	#Adding an activation function

            #Adding the dropout layer
            model.add(Dropout(0.5))  			#Adding the dropout layer reduces the overfitting of the model
            
            #Adding the output Dense layer of 4 Nodes. (4 Classes)
            model.add(Dense(4))

            #Adding and 'Softmax' activation function
            model.add(Activation('softmax'))

            #Adding the Compile methond since we have multi-class classifcation we will use learning methond as 'rmsprop'  
            #loss funtion as 'Categorical_crossentropy' and using metrics as 'accuracy'
            model.compile(loss='categorical_crossentropy',
                           optimizer = 'rmsprop',
                           metrics=['accuracy'])
            
            #Fitting data set into the model using Fit_generator function
            #Fit generator is used int his model due to data augmentation(ImageDataGenerator)  
            #We are using 50 Epochs for training were 200 steps are done per epoch and each step will be having 32 batch of images.
            results = model.fit_generator(train_image_gen,epochs =50,steps_per_epoch=200,validation_data=test_image_gen,validation_steps=12)

            #Printing the summary of the model
            print(model.summary())
            #Saving the Model with respective Model name.
            model.save(Name+'.h5')





#Printing class and their Indices
print(train_image_gen.class_indices)






