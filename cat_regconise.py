import numpy as np
from PIL import Image
from L_layers_nn import *

image_name = input("Image's name: ")
label_image = input("0 if non-cat, 1 if cat: ")
label_image_y = [int(label_image)]
fname = "images/" + image_name
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
image = image/255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(image, label_image_y, parameters)
print("y = " + str(np.squeeze(label_image_y)) + ", your L-layer model predicts a\"" + classes[int(np.squeeze(label_image_y)),].decode("utf-8") + "\" picture.")