import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import math

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 5

train_datagen_cls = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)

train_datagen_seg = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

labels_df = pd.read_csv('dataset/classification_labels.csv')
labels_df['class'] = labels_df['class'].astype(str)
class_names = sorted(labels_df['class'].unique())
class_indices = dict((c, i) for i, c in enumerate(class_names))
labels_df['class_idx'] = labels_df['class'].map(class_indices)

train_df, val_df = train_test_split(labels_df, test_size=0.2, stratify=labels_df['class'], random_state=42)

train_generator_cls = train_datagen_cls.flow_from_dataframe(
    dataframe=train_df,
    directory='dataset/images/',
    x_col='filename',
    y_col='class',
    subset='training',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator_cls = train_datagen_cls.flow_from_dataframe(
    dataframe=val_df,
    directory='dataset/images/',
    x_col='filename',
    y_col='class',
    subset='validation',
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

def segmentation_generator(image_dir, mask_dir, dataframe, batch_size, img_size, augmentor):
    image_generator = augmentor.flow_from_dataframe(
        dataframe=dataframe,
        directory=image_dir,
        x_col='filename',
        class_mode=None,
        target_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    mask_generator = augmentor.flow_from_dataframe(
        dataframe=dataframe,
        directory=mask_dir,
        x_col='filename',
        class_mode=None,
        color_mode='grayscale',
        target_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    while True:
        img = image_generator.next()
        mask = mask_generator.next()
        mask = np.where(mask > 0.5, 1, 0)
        yield img, mask

train_generator_seg = segmentation_generator(
    image_dir='dataset/images/',
    mask_dir='dataset/segmentation_masks/',
    dataframe=train_df,
    batch_size=BATCH_SIZE,
    img_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    augmentor=train_datagen_seg
)

validation_generator_seg = segmentation_generator(
    image_dir='dataset/images/',
    mask_dir='dataset/segmentation_masks/',
    dataframe=val_df,
    batch_size=BATCH_SIZE,
    img_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    augmentor=train_datagen_seg
)

base_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3))
base_model.trainable = False
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model_cls = models.Model(inputs=base_model.input, outputs=predictions)
model_cls.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

def unet_model(input_size=(224, 224, 3)):
    inputs = layers.Input(input_size)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

model_seg = unet_model()
model_seg.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

callbacks_cls = [
    EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('best_model_cls.h5', save_best_only=True, monitor='val_loss')
]

callbacks_seg = [
    EarlyStopping(patience=10, monitor='val_loss', restore_best_weights=True),
    ModelCheckpoint('best_model_seg.h5', save_best_only=True, monitor='val_loss')
]

steps_per_epoch_cls = math.ceil(train_generator_cls.n / train_generator_cls.batch_size)
validation_steps_cls = math.ceil(validation_generator_cls.n / validation_generator_cls.batch_size)

history_cls = model_cls.fit(
    train_generator_cls,
    steps_per_epoch=steps_per_epoch_cls,
    epochs=50,
    validation_data=validation_generator_cls,
    validation_steps=validation_steps_cls,
    callbacks=callbacks_cls
)

steps_per_epoch_seg = math.ceil(len(train_df) / BATCH_SIZE)
validation_steps_seg = math.ceil(len(val_df) / BATCH_SIZE)

history_seg = model_seg.fit(
    train_generator_seg,
    steps_per_epoch=steps_per_epoch_seg,
    epochs=50,
    validation_data=validation_generator_seg,
    validation_steps=validation_steps_seg,
    callbacks=callbacks_seg
)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False
model_cls.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

history_finetune_cls = model_cls.fit(
    train_generator_cls,
    steps_per_epoch=steps_per_epoch_cls,
    epochs=20,
    validation_data=validation_generator_cls,
    validation_steps=validation_steps_cls,
    callbacks=callbacks_cls
)

model_seg.compile(optimizer=optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])

history_finetune_seg = model_seg.fit(
    train_generator_seg,
    steps_per_epoch=steps_per_epoch_seg,
    epochs=20,
    validation_data=validation_generator_seg,
    validation_steps=validation_steps_seg,
    callbacks=callbacks_seg
)

model_cls.load_weights('best_model_cls.h5')
validation_generator_cls.reset()
preds = model_cls.predict(validation_generator_cls, steps=validation_steps_cls, verbose=1)
pred_classes = np.argmax(preds, axis=1)
true_classes = validation_generator_cls.classes[:len(pred_classes)]
class_labels = list(validation_generator_cls.class_indices.keys())
report = classification_report(true_classes, pred_classes, target_names=class_labels)
print(report)
cm = confusion_matrix(true_classes, pred_classes)
print(cm)

model_seg.load_weights('best_model_seg.h5')

def plot_segmentation(model, generator, steps=1):
    for i in range(steps):
        images, masks = next(generator)
        preds = model.predict(images)
        preds = (preds > 0.5).astype(np.float32)
        for j in range(len(images)):
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(images[j])
            plt.title('Input Image')
            plt.subplot(1, 3, 2)
            plt.imshow(masks[j].squeeze(), cmap='gray')
            plt.title('True Mask')
            plt.subplot(1, 3, 3)
            plt.imshow(preds[j].squeeze(), cmap='gray')
            plt.title('Predicted Mask')
            plt.show()

plot_segmentation(model_seg, validation_generator_seg, steps=3)
results_seg = model_seg.evaluate(validation_generator_seg, steps=validation_steps_seg)
print(f"Segmentation Model Loss: {results_seg[0]}")
print(f"Segmentation Model Accuracy: {results_seg[1]}")
print(f"Mean IoU: {results_seg[2]}")
