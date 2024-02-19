# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2 
if tf.config.experimental.list_physical_devices('GPU'):
    print("TensorFlow is using GPU!")
else:
    print("TensorFlow is using CPU!")
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import cv2
import shutil
import time
from sklearn.metrics import classification_report

# Define paths for datasets and results based on local directory
train_csv = "C:\\Users\\mikec\\Downloads\\butterfly_mobilemodel\\Training_set.csv"
train_folder = "C:\\Users\\mikec\\Downloads\\butterfly_mobilemodel\\train"
test_csv = "C:\\Users\\mikec\\Downloads\\butterfly_mobilemodel\\Testing_set.csv"
test_folder = "C:\\Users\\mikec\\Downloads\\butterfly_mobilemodel\\test"

# For results, you can create a 'results' folder under the same directory or choose another location
result_path = f"C:\\Users\\mikec\\Downloads\\butterfly_mobilemodel\\result"
os.makedirs(result_path, exist_ok=True)

# Paths for saving results
checkpoint_path = os.path.join(result_path, "best_model.h5")
loss_image_path = os.path.join(result_path, 'validation loss.png')
acc_image_path = os.path.join(result_path, 'validation accuracy.png')
confusion_image_path = os.path.join(result_path, 'confusion matrix.png')


# Load the datasets
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

# Visualize the distribution of butterfly categories
plt.figure(figsize=(15, 7))
train_df['label'].value_counts().plot(kind='bar')
plt.title('Distribution of Butterfly Categories')
plt.xlabel('Butterfly Category')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Set parameters for data preprocessing and model training
image_size = (150, 150)
batch_size = 32
epochs = 100
learning_rate = 0.0001
class_name = list(set(train_df['label']))
print(class_name)

# Preprocess images: load, resize, normalize, and assign labels
features = []
labels = []
for image in tqdm(os.listdir(train_folder), desc="Preprocess Image"):
    label_name = train_df.loc[train_df['filename'] == image, 'label'].values[0]
    label = class_name.index(label_name)
    image_read = cv2.imread(os.path.join(train_folder, image))
    image_resized = cv2.resize(image_read, image_size)
    image_normalized = image_resized / 255.0
    features.append(image_normalized)
    labels.append(label)

# Convert lists to numpy arrays
features = np.asarray(features)
labels = np.asarray(labels)

# Split data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True, random_state=42)

# Free up memory
del features
del labels

# Set up the model using MobileNetV2 as the base model for transfer learning
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(image_size[0], image_size[1], 3)
)
num_layers_to_train = int(np.ceil(0.2 * len(base_model.layers)))
for layer in base_model.layers[:num_layers_to_train]:
    layer.trainable = False
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu', kernel_regularizer='l2')(x)
predictions = Dense(75, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

# Compile and train the model
model.compile(optimizer=Adam(learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True)
start_time = time.time()
history = model.fit(
    X_train,
    y_train,
    epochs=epochs,
    validation_data=(X_valid, y_valid),
    callbacks=[model_checkpoint, early_stopping],
    batch_size=batch_size
)
end_time = time.time()
print("Training Time", end_time - start_time)

# Plot training and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(loss_image_path)
plt.show()

# Plot training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(acc_image_path)
plt.show()

# Evaluate the model on the test set and generate a classification report
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
classification_rep = classification_report(y_test, y_pred, target_names=class_name, digits=4)
print("Classification Report:\n", classification_rep)
classification_file = 'classification_report.txt'
with open(classification_file, 'w') as file:
    file.write(classification_rep)
