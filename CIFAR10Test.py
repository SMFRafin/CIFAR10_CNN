import tensorflow as tf
from tensorflow import keras
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical
from keras.utils import plot_model
import visualkeras
from keras.layers import Conv2D
from collections import defaultdict
loaded_model=keras.models.load_model("CIFAR10.model")

(X_test,y_test)=cifar10.load_data()[1]
X_test=X_test/255.0
y_test=to_categorical(y_test,10)

predictions=loaded_model.predict(X_test)
test_loss,test_accuracy=loaded_model.evaluate(X_test,y_test)

print("Test Loss:",test_loss)
print("Test Accuracy:",test_accuracy)

color_map=defaultdict(dict)
color_map[Conv2D]['fill']='orange'

plot_model(loaded_model,to_file='CIFAR10.png',show_shapes=True,show_layer_names=True)
visualkeras.layered_view(loaded_model,legend=True,color_map=color_map,to_file='CIFAR103D.png')

num_images_to_plot=10
indices=np.random.choice(range(len(X_test)), num_images_to_plot,replace=False)

y_test=np.argmax(y_test,axis=1)
plt.figure(figsize=(15, 8))
for i, index in enumerate(indices,1):
    plt.subplot(2,num_images_to_plot//2,i)
    plt.imshow(X_test[index])
    plt.title(f"Actual:{y_test[index]}\nPredicted: {np.argmax(predictions[index])}")
    plt.axis("off") 
plt.show()
