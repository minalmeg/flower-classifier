import tensorflow.compat.v1 as tf
import numpy as np
import os,glob,cv2
import sys,argparse
tf.disable_v2_behavior()

# First, pass the path of the image
dir_path = os.path.dirname(os.path.realpath(__file__))
image_path=sys.argv[1]
filename = dir_path +'/' +image_path
image_size=128
num_channels=3
images = []
# Reading the image using OpenCV
image = cv2.imread(filename)
image1 = cv2.imread(filename)
# Resizing the image to our desired size and preprocessing will be done exactly as done during training
image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0)
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(1, image_size,image_size,num_channels)

## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('base_model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, len(os.listdir('training_data'))))


### Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_rose probability_of_sunflower]
maxElement = np.max(result)
l = np.argmax(result, axis=1)
'''
if l == 0:
    print("Acaulis")
elif l == 1:
    print("Daisy")
elif l == 2:
    print("Dandelion")
elif l == 3:
    print("Frangipani")
elif l == 4:
    print("Hibiscus")
elif l == 5:
    print("Lotus")
elif l == 6:
    print("Petunia")
elif l == 7:
    print("Rose")
elif l == 8:
    print("Sunflower")
elif l == 9:
    print("Tulip")
'''

if l == 0:
    string = "Acaulis"
elif l == 1:
    string = "Daisy"
elif l == 2:
    string = "Dandelion"
elif l == 3:
    string = "Frangipani"
elif l == 4:
    string = "Hibiscus"
elif l == 5:
    string = "Lotus"
elif l == 6:
    string = "Petunia"
elif l == 7:
    string = "Rose"
elif l == 8:
    string = "Sunflower"
elif l == 9:
    string = "Tulip"

cv2.putText(image1, string, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
cv2.imshow("img",image1)
cv2.waitKey(0)
cv2.destroyAllWindows()