#!/usr/bin/env python3
"""
Rice Leaf Disease Detection — Model Training Script
====================================================
Uses MobileNetV2 (transfer learning) on the Kaggle Rice Leaf Disease dataset.

Dataset: https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases
Expected directory structure after extraction:

  dataset/
  ├── Bacterial leaf blight/   (or "Bacterial Blight")
  ├── Brown spot/
  ├── Leaf smut/               (mapped to Leaf Blast in this project)
  └── (optional) Healthy/

Adjust DATASET_DIR and CLASS_NAMES below if your folder names differ.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D,
                                      Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                         ReduceLROnPlateau)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ─── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR = 'dataset'          # path to your dataset folder
MODEL_SAVE_PATH = 'model/rice_disease_model.h5'
LABELS_SAVE_PATH = 'model/class_labels.json'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_FROZEN = 10               # epochs with base model frozen
EPOCHS_FINETUNE = 10             # epochs for fine-tuning
LEARNING_RATE = 1e-3
FINETUNE_LR = 1e-5
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
RANDOM_SEED = 42

# Map raw folder names → display class names
FOLDER_TO_CLASS = {
    'Bacterial leaf blight': 'Bacterial Blight',
    'bacterial leaf blight': 'Bacterial Blight',
    'Bacterial Blight': 'Bacterial Blight',
    'Brown spot': 'Brown Spot',
    'brown spot': 'Brown Spot',
    'Brown Spot': 'Brown Spot',
    'Leaf smut': 'Leaf Blast',     # dataset uses "Leaf smut" label
    'leaf smut': 'Leaf Blast',
    'Leaf Blast': 'Leaf Blast',
    'Healthy': 'Healthy',
    'healthy': 'Healthy',
}

# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_dataset(dataset_dir):
    """Walk dataset directory and return (paths, labels)."""
    print(f"[INFO] Loading dataset from: {dataset_dir}")
    image_paths, labels = [], []
    class_names_set = set()

    for folder in sorted(os.listdir(dataset_dir)):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        class_name = FOLDER_TO_CLASS.get(folder, folder)
        class_names_set.add(class_name)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(folder_path, fname))
                labels.append(class_name)

    class_names = sorted(class_names_set)
    print(f"[INFO] Found {len(image_paths)} images across {len(class_names)} classes: {class_names}")
    return np.array(image_paths), np.array(labels), class_names


def build_model(num_classes):
    """Build MobileNetV2-based transfer learning model."""
    base_model = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # freeze initially

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    return model, base_model


def get_data_generators(X_train, y_train, X_val, y_val, X_test, y_test,
                         class_names, batch_size):
    """Create tf.data pipelines for train/val/test."""
    label_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)

    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = img / 255.0
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, 0.0, 1.0)
        return img, label

    def encode_label(path, label_str):
        label_idx = tf.py_function(
            lambda l: label_to_idx[l.numpy().decode()], [label_str], tf.int32)
        label_one_hot = tf.one_hot(label_idx, num_classes)
        return path, label_one_hot

    def make_dataset(paths, labels, augment_data=False, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(encode_label, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(lambda p, l: (load_and_preprocess(p, l)), num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=len(paths), seed=RANDOM_SEED)
        if augment_data:
            ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = make_dataset(X_train, y_train, augment_data=True, shuffle=True)
    val_ds   = make_dataset(X_val,   y_val)
    test_ds  = make_dataset(X_test,  y_test)
    return train_ds, val_ds, test_ds


def plot_history(history, suffix=''):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric in zip(axes, ['accuracy', 'loss']):
        ax.plot(history.history[metric], label=f'Train {metric}')
        ax.plot(history.history[f'val_{metric}'], label=f'Val {metric}')
        ax.set_title(f'Model {metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.legend()
    plt.tight_layout()
    out = f'model/training_history{suffix}.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] Saved training plot → {out}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    out = 'model/confusion_matrix.png'
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] Saved confusion matrix → {out}")

# ─── Main Training Loop ────────────────────────────────────────────────────────

def train():
    os.makedirs('model', exist_ok=True)

    if not os.path.exists(DATASET_DIR):
        print(f"[ERROR] Dataset directory '{DATASET_DIR}' not found.")
        print("        Download the Rice Leaf Disease dataset from Kaggle:")
        print("        https://www.kaggle.com/datasets/vbookshelf/rice-leaf-diseases")
        print("        Extract it so that 'dataset/' sits next to train_model.py")
        return

    # 1. Load data
    image_paths, labels, class_names = load_dataset(DATASET_DIR)
    print(f"[INFO] Class distribution:")
    for c in class_names:
        print(f"        {c}: {np.sum(labels == c)} images")

    # Save class labels
    with open(LABELS_SAVE_PATH, 'w') as f:
        json.dump(class_names, f)
    print(f"[INFO] Class labels saved → {LABELS_SAVE_PATH}")

    num_classes = len(class_names)

    # 2. Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        image_paths, labels, test_size=TEST_SPLIT, stratify=labels, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SPLIT / (1 - TEST_SPLIT),
        stratify=y_temp, random_state=RANDOM_SEED)

    print(f"[INFO] Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 3. Build data pipelines
    train_ds, val_ds, test_ds = get_data_generators(
        X_train, y_train, X_val, y_val, X_test, y_test, class_names, BATCH_SIZE)

    # 4. Build model
    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    # ── Phase 1: Train classification head (frozen base) ──
    print("\n[PHASE 1] Training classification head (base model frozen)...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FROZEN,
        callbacks=callbacks
    )
    plot_history(history1, suffix='_phase1')

    # ── Phase 2: Fine-tune last 30 layers of MobileNetV2 ──
    print("\n[PHASE 2] Fine-tuning last 30 layers of MobileNetV2...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=FINETUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINETUNE,
        callbacks=callbacks
    )
    plot_history(history2, suffix='_phase2')

    # 5. Evaluate on test set
    print("\n[INFO] Evaluating on test set...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"[RESULT] Test Accuracy: {test_acc*100:.2f}%  |  Test Loss: {test_loss:.4f}")

    # 6. Classification report + confusion matrix
    y_true, y_pred = [], []
    label_to_idx = {n: i for i, n in enumerate(class_names)}
    for imgs, lbls in test_ds:
        preds = model.predict(imgs, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(lbls.numpy(), axis=1))

    print("\n[RESULT] Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    plot_confusion_matrix(y_true, y_pred, class_names)

    print(f"\n[DONE] Model saved → {MODEL_SAVE_PATH}")
    print("       Start the web app:  python app.py")


if __name__ == '__main__':
    train()
