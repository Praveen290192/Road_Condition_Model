#Importing the Modules and classes
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

#Loading the model
#Location of the Directory Saved Model Directory
test_model = load_model('D:/images/Model_and_files/Saved_Model/road_2_D_O_C_32.h5')


#Loading image
#Location of the Directory for testing single images
test_img ='D:/images/Model_and_files/Extra_images_with_no_label/images4.jpg'

#Image Preprocessing
test_img = image.load_img(test_img,target_size=(150,150)) 		#loading the image in the test_img by changing it's size, since we trained on 150x150 size images.
test_img= image.img_to_array(test_img)							#Changing the image to array
test_img=np.expand_dims(test_img,axis=0)						#Adding extra dimension so as to indicate the batch size
test_img = test_img/255											#Changing the pixels values between 0 & 1

#Predicting the class
class_pred = test_model.predict_classes(test_img)

#Printing output
classes =['clean', 'dirty', 'snow', 'wet'] 						#defining list of classes with there suitable indices

print('Test image is predicted as '+classes=[class_pred[0]]) 	#Printing out the predictions