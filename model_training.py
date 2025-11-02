# model_training.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks, optimizers

# Configuration
DATA_DIR = 'datasets/labeled'  # structure: datasets/labeled/happy/*.jpg etc
IMG_SIZE = (48,48)
BATCH_SIZE = 64
EPOCHS = 30
MODEL_SAVE_PATH = 'models/emotion_model.h5'

def build_model(input_shape=(48,48,1), num_classes=7):
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32,(3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64,(3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64,(3,3), activation='relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128,(3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    model.compile(optimizer=optimizers.Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train():
    if not os.path.exists(DATA_DIR):
        raise SystemExit(f"Dataset directory '{DATA_DIR}' not found. Please create and add labeled images before training.")
    datagen = ImageDataGenerator(rescale=1./255,
                                 validation_split=0.2,
                                 rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 horizontal_flip=True)
    train_gen = datagen.flow_from_directory(DATA_DIR,
                                            target_size=IMG_SIZE,
                                            color_mode='grayscale',
                                            batch_size=BATCH_SIZE,
                                            class_mode='categorical',
                                            subset='training')
    val_gen = datagen.flow_from_directory(DATA_DIR,
                                          target_size=IMG_SIZE,
                                          color_mode='grayscale',
                                          batch_size=BATCH_SIZE,
                                          class_mode='categorical',
                                          subset='validation')

    num_classes = train_gen.num_classes
    model = build_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1), num_classes=num_classes)

    cb = [
        callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_accuracy', mode='max'),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    ]

    model.fit(train_gen,
              epochs=EPOCHS,
              validation_data=val_gen,
              callbacks=cb)
    print("Training finished. Model saved to", MODEL_SAVE_PATH)

if __name__ == '__main__':
    os.makedirs('models', exist_ok=True)
    train()
