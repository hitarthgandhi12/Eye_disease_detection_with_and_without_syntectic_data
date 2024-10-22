"""CNN. RESNET, EFFICIENT NET With GANs"""


import tensorflow as tf
from tensorflow.keras import layers

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

# Generator Model
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((16, 16, 256)),

        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Discriminator Model
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[256, 256, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = build_generator()
discriminator = build_discriminator()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# GAN model for training
@tf.function
def train_step(images, batch_size=32):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Train GAN
def train_gan(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

# Function to generate synthetic images using GAN
def generate_gan_images(generator, num_images):
    noise = tf.random.normal([num_images, 100])  # Assuming the input to your GAN is noise of shape [100]
    generated_images = generator(noise, training=False)
    return generated_images

import numpy as np

# Adjust original dataset to 40% of the original size for training
train_original_ds = tf.keras.utils.image_dataset_from_directory(
    directory = 'C:/Users/cl502_05/Downloads/NN_Eyes_disease_detection-20240930T053028Z-001/NN_Eyes_disease_detection/dataset',
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode='categorical',
)

# Get 40% of the training data
total_training_samples = len(train_original_ds)  # Get total number of samples
num_original_samples_for_train = int(total_training_samples * 0.4)

# Shuffle and reduce dataset size to 40% for training
train_original_ds = train_original_ds.take(num_original_samples_for_train)

# Load GAN-generated images (40% of total training size)
num_gan_images = num_original_samples_for_train  # Same number as 40% of original images

# Generate synthetic images and corresponding labels
gan_images = generate_gan_images(generator, num_gan_images)  # Generating GAN images (assuming GAN is pre-trained)
gan_labels = np.zeros((num_gan_images, 4))  # Assuming 4 classes, label GAN images as required

# Preprocess GAN images (rescale them to [0,1] range)
gan_images_rescaled = gan_images * 0.5 + 0.5  # Adjust range from [-1, 1] to [0, 1] if using 'tanh' in GAN generator

# Function to resize GAN-generated images to match the original dataset shape
def resize_gan_images(gan_images, target_size=(256, 256)):
    resized_images = tf.image.resize(gan_images, target_size)
    return resized_images

# Resize GAN-generated images to (256, 256)
gan_images_rescaled_resized = resize_gan_images(gan_images_rescaled, target_size=(256, 256))

# Combine original and GAN-generated datasets into one training set
def combine_datasets(original_dataset, gan_images, gan_labels):
    # Convert original dataset to a list for manipulation
    original_images, original_labels = [], []

    for batch_images, batch_labels in original_dataset:
        original_images.append(batch_images.numpy())
        original_labels.append(batch_labels.numpy())

    # Flatten the list of batches into single numpy arrays
    original_images = np.concatenate(original_images, axis=0)
    original_labels = np.concatenate(original_labels, axis=0)

    # Combine original and resized GAN images
    combined_images = np.concatenate([original_images, gan_images], axis=0)
    combined_labels = np.concatenate([original_labels, gan_labels], axis=0)

    # Shuffle combined dataset
    combined_dataset = tf.data.Dataset.from_tensor_slices((combined_images, combined_labels))
    combined_dataset = combined_dataset.shuffle(len(combined_images)).batch(32)  # Adjust batch size as required

    return combined_dataset

# Create mixed training dataset
train_mixed_ds = combine_datasets(train_original_ds, gan_images_rescaled_resized, gan_labels)


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
    print("RESNET GANs")

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
    print("EFFICIENTNET GANs")

    model = tf.keras.Sequential([  # Fixed this line
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    return model

# Compile model
from sklearn.metrics import classification_report

# Function to compile model with additional metrics
def compile_model(model):
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy', 
                            tf.keras.metrics.Precision(), 
                            tf.keras.metrics.Recall()])
    return model


# Train model
def train_model(model, train_mixed_ds, validation_ds):
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    history = model.fit(train_mixed_ds, validation_data=validation_ds, epochs=1, callbacks=[early_stopping])  # Increased epochs for better training
    return history

# Evaluate model with additional metrics
def evaluate_model(model, dataset):
    loss, accuracy, precision, recall = model.evaluate(dataset)
    print(f"Model: {model.name}")
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    # Get predictions to compute F1 score
    y_true = np.concatenate([y.numpy() for _, y in dataset], axis=0)
    y_pred = model.predict(dataset)
    
    # Convert predictions to class labels
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)

    # Print classification report
    print(classification_report(y_true_classes, y_pred_classes))

# Train and evaluate Original CNN
cnn_model = create_cnn_model()
compile_model(cnn_model)
train_model(cnn_model, train_mixed_ds, validation_ds)
evaluate_model(cnn_model, validation_ds)

# Train and evaluate ResNet50
resnet_model = create_resnet_model()
compile_model(resnet_model)
train_model(resnet_model, train_mixed_ds, validation_ds)
evaluate_model(resnet_model, validation_ds)

# Train and evaluate EfficientNetB0
efficientnet_model = create_efficientnet_model()
compile_model(efficientnet_model)
train_model(efficientnet_model, train_mixed_ds, validation_ds)
evaluate_model(efficientnet_model, validation_ds)


# Evaluate each model
for model in [cnn_model, resnet_model, efficientnet_model]:
    evaluate_model(model, validation_ds)



# Load original training dataset (without GANs)
original_train_ds = tf.keras.utils.image_dataset_from_directory(
    directory = 'C:/Users/cl502_05/Downloads/NN_Eyes_disease_detection-20240930T053028Z-001/NN_Eyes_disease_detection/dataset',
    batch_size=32,
    image_size=(256, 256),
    validation_split=0.2,
    subset="training",
    seed=123,
    label_mode='categorical',
)

# Preprocess original training dataset (rescale)
original_train_ds = original_train_ds.map(lambda x, y: (rescale(x), y))

# Train the model on the original dataset
history_original = model.fit(original_train_ds, epochs=15, validation_data=validation_ds)

# Train the model
history_gan = model.fit(train_mixed_ds, epochs=15, validation_data=validation_ds)

print("GAN Model Validation Accuracy:", history_gan.history['val_accuracy'])
print("Original Model Validation Accuracy:", history_original.history['val_accuracy'])

