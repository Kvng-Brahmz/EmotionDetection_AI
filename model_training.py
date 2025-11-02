import os
import argparse
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from pathlib import Path

MODEL_OUT = "models"
Path(MODEL_OUT).mkdir(exist_ok=True, parents=True)
OUTPUT_MODEL = os.path.join(MODEL_OUT, "emotion_model.h5")

def build_model(input_shape=(48,48,1), num_classes=7):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def main(args):
    train_dir = args.train_dir
    val_dir = args.val_dir

    train_gen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_flow = train_gen.flow_from_directory(train_dir, target_size=(48,48), color_mode='grayscale',
                                               batch_size=args.batch_size, class_mode='categorical')
    val_flow = val_gen.flow_from_directory(val_dir, target_size=(48,48), color_mode='grayscale',
                                           batch_size=args.batch_size, class_mode='categorical')

    model = build_model(input_shape=(48,48,1), num_classes=train_flow.num_classes)
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])
    callbacks = [
        ModelCheckpoint(OUTPUT_MODEL, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
    ]
    model.fit(train_flow, epochs=args.epochs, validation_data=val_flow, callbacks=callbacks)
    print("Training done. Model saved at:", OUTPUT_MODEL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="datasets/train", help="training directory with subfolders per class")
    parser.add_argument("--val_dir", default="datasets/val", help="validation directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
