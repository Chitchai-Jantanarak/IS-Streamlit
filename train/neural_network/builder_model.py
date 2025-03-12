import tensorflow as tf
import pathlib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization, Dense, Dropout, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.metrics import Recall, Precision

def load_image(base_path) -> tuple:
    """
    Load images from the specified directory
    
    Args:
        base_path (str): Base path to the image directory [INNER : {PreData : test, train, val}]
    
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """

    data_dir = pathlib.Path(base_path)

    if not data_dir.exists():
        raise ValueError(f"Director does not exist: {data_dir}")
    
    train_x = tf.keras.utils.image_dataset_from_directory(
        str(data_dir / 'train'),
        color_mode='rgb',  # Changed from 'grayscale' to 'rgb'
        seed=123,
        image_size=(224, 224),  # Resize images as const
        batch_size=32
    )

    test_x = tf.keras.utils.image_dataset_from_directory(
        str(data_dir / 'test'),
        color_mode='rgb',  # Changed from 'grayscale' to 'rgb'
        seed=123,
        image_size=(224, 224),  # Resize images as const
        batch_size=32
    )
    
    val_x = tf.keras.utils.image_dataset_from_directory(
        str(data_dir / 'val'),
        color_mode='rgb',  # Changed from 'grayscale' to 'rgb'
        seed=123,
        image_size=(224, 224),  # Resize images as const
        batch_size=32
    )

    return (train_x, test_x, val_x)

def preprocess_image(train_x, test_x, val_x) -> tuple:
    """
    Preprocess rescaling an image bit in range [0, 1] 
    
    Args:
        train_x (Array)
        test_x (Array)
        val_x (Array)
    
    Returns:
        Tuple of (train_dataset, validation_dataset)
    """
    # Augment Filtering
    datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    def normalize(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    # Normalize datasets
    train_x_norm = train_x.map(normalize)
    test_x_norm = test_x.map(normalize)
    val_x_norm = val_x.map(normalize)

    return (train_x_norm, test_x_norm, val_x_norm)

def modelsConstruct() -> tf.keras.models.Sequential:
    """
    Model builder using VGG16 as base model
    """
    base_model = VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3)  # Match image size and RGB channels
    )

    model = Sequential([
        base_model,
        Flatten(),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    base_model.trainable = False

    return model

def modelsBuilt(model):
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', Recall(), Precision()]
    )

    model.summary()
    return model

def training(train_data, val_data, model):
    # Add early stopping and model checkpoint
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'pneumonia_model.h5', 
        monitor='val_accuracy', 
        save_best_only=True
    )

    model.fit(
        train_data,
        epochs=10,
        validation_data=val_data,
        callbacks=[early_stopping, model_checkpoint]
    )

def main():
    base_path = '../../data/neural_network/chest_xray'

    try:
        train_x, test_x, val_x = load_image(base_path)
        train_x_prep, test_x_prep, val_x_prep = preprocess_image(train_x, test_x, val_x)
        
        # Print dataset information for debugging
        for dataset, name in [(train_x_prep, 'Train'), (test_x_prep, 'Test'), (val_x_prep, 'Validation')]:
            for images, labels in dataset.take(1):
                print(f"{name} Dataset:")
                print(f"Images shape: {images.shape}")
                print(f"Labels shape: {labels.shape}")
        
        model = modelsConstruct()
        model = modelsBuilt(model)
        training(train_x_prep, val_x_prep, model)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()