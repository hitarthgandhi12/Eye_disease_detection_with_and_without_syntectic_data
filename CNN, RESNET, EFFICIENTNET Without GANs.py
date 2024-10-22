#CNN, RESNET, EFFICIENTNET Without GANs

import tensorflow as tf
import tensorflow as tf
layers = tf.keras.layers

import numpy as np

# Define rescaling layer
rescale = tf.keras.layers.Rescaling(1./255)

# Load train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory='C:/Users/cl502_05/Downloads/NN_Eyes_disease_detection-20240930T053028Z-001/NN_Eyes_disease_detection/dataset',
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode='categorical',
)

# Preprocess train dataset (rescale)
train_ds = train_ds.map(lambda x, y: (rescale(x), y))

# Load validation dataset
validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory='C:/Users/cl502_05/Downloads/NN_Eyes_disease_detection-20240930T053028Z-001/NN_Eyes_disease_detection/dataset',
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.2,
    subset="validation",
    seed=123,
    label_mode='categorical',
)

# Preprocess validation dataset (rescale)
validation_ds = validation_ds.map(lambda x, y: (rescale(x), y))

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    directory='C:/Users/cl502_05/Downloads/NN_Eyes_disease_detection-20240930T053028Z-001/NN_Eyes_disease_detection/dataset',
    batch_size=32,
    image_size=(256, 256),
    label_mode='categorical',
    shuffle=False,
)

# Preprocess test dataset (rescale)
test_ds = test_ds.map(lambda x, y: (rescale(x), y))

# Check the first image shape in the training dataset
print("Shape of the first image in the training dataset:", next(iter(train_ds))[0][0].shape)

# Define CNN model
def create_cnn_model():
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    return model

# Function to create a ResNet model
def create_resnet_model():
    base_model = tf.keras.applications.ResNet50(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    print("RESNET")

    model = tf.keras.Sequential([  # Fixed this line
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    return model

# Function to create an EfficientNet model
def create_efficientnet_model():
    base_model = tf.keras.applications.EfficientNetB0(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    print("EFFICIENTNET")

    model = tf.keras.Sequential([  # Fixed this line
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    return model

# Compile model
def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train model
def train_model(model, train_ds, validation_ds):
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=validation_ds, epochs=15, callbacks=[early_stopping])  # Increased epochs for better training
    return history

# Model 1: Original CNN
cnn_model = create_cnn_model()
compile_model(cnn_model)
history_cnn = train_model(cnn_model, train_ds, validation_ds)



# Model 2: ResNet50
resnet_model = create_resnet_model()
compile_model(resnet_model)
history_resnet = train_model(resnet_model, train_ds, validation_ds)

# Model 3: EfficientNetB0
efficientnet_model = create_efficientnet_model()
compile_model(efficientnet_model)
history_efficientnet = train_model(efficientnet_model, train_ds, validation_ds)

# Evaluate model
for model in [cnn_model, resnet_model, efficientnet_model]:
    loss, accuracy = model.evaluate(validation_ds)
    print(f"{model.name} - Loss: {loss}, Accuracy: {accuracy}")
