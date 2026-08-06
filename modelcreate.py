"""
Clash Royale Character Classifier - TensorFlow
==============================================
Images natives : 64 (H) x 52 (W) pixels

Usage :
  python Ia.py --data_dir "C:/Users/anima/Documents/GitHub/Insta_Proxy/image_loack"
  python Ia.py --predict "chemin/vers/image.png"
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import time

# ─── Paramètres ───────────────────────────────────────────────────────────────
IMG_H            = 117#64#
IMG_W            = 138#52
BATCH_SIZE       = 8
EPOCHS           = 500         # Beaucoup plus d'epochs
STEPS_PER_EPOCH  = 100         # 100 batchs générés par epoch (augmentation intensive)
MODEL_PATH       = "model/clash_3D_board_classifiersaison9.keras"
LABELS_PATH      = "model/clash_3D_board_classifiersiason9.json"
CHECKPOINT_DIR   = "model/checkpoints"
CHECKPOINT_PATH  = os.path.join(
    CHECKPOINT_DIR,
    "clash_{epoch:03d}_{accuracy:.4f}.keras",
)
# ──────────────────────────────────────────────────────────────────────────────


def build_model(num_classes: int) -> tf.keras.Model:
    model = models.Sequential([
        layers.Input(shape=(IMG_H, IMG_W, 3)),

        # Bloc 1
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),

        # Bloc 2
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),

        # Bloc 3
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_generator(data_dir: str):
    """Augmentation très agressive pour compenser le manque de données."""
    gen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.6, 1.4],
        channel_shift_range=30.0,
        fill_mode="nearest",
    )
    ds = gen.flow_from_directory(
        data_dir,
        target_size=(IMG_H, IMG_W),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    return ds


def save_model_safely(model: tf.keras.Model, target_path: str) -> str:
    """Save to a temporary file, then replace the target if Windows allows it."""
    model_dir = os.path.dirname(target_path) or "."
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    stamp = int(time.time())
    tmp_path = os.path.join(model_dir, f".tmp_{stamp}_{os.path.basename(target_path)}")
    model.save(tmp_path)

    try:
        os.replace(tmp_path, target_path)
        print(f"Modele final sauvegarde : {target_path}")
        return target_path
    except OSError as exc:
        fallback_path = os.path.join(CHECKPOINT_DIR, f"clash_final_{stamp}.keras")
        os.replace(tmp_path, fallback_path)
        print(f"Impossible de remplacer {target_path} : {exc}")
        print(f"Modele final sauvegarde ici : {fallback_path}")
        return fallback_path


def train(data_dir: str):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_ds = get_generator(data_dir)
    num_classes = len(train_ds.class_indices)
    n_images = train_ds.samples

    labels = {str(v): k for k, v in train_ds.class_indices.items()}
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    model = build_model(num_classes)
    model.summary()

    callbacks = [
        # Pas d'EarlyStopping agressif — on veut entraîner longtemps
        tf.keras.callbacks.EarlyStopping(
            patience=80, restore_best_weights=True, monitor="accuracy"
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=20, monitor="loss", verbose=1, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            CHECKPOINT_PATH,
            save_best_only=True,
            monitor="accuracy",
            mode="max",
            verbose=0,
        ),
    ]

    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=callbacks,
    )

    save_model_safely(model, MODEL_PATH)
    loss, acc = model.evaluate(train_ds, verbose=0)



def predict(image_path: str) -> str:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}\n"
            "Lance d'abord : python Ia.py --data_dir <chemin>"
        )
    with open(LABELS_PATH) as f:
        labels = json.load(f)

    model = tf.keras.models.load_model(MODEL_PATH)

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMG_H, IMG_W))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)

    preds = model.predict(arr, verbose=0)[0]
    top3  = np.argsort(preds)[::-1][:3]

    for rank, idx in enumerate(top3, 1):
        name  = labels[str(idx)]
        score = preds[idx] * 100
        marker = " <-- prediction" if rank == 1 else ""
        print(f"  {rank}. {name:<25} {score:5.1f}%{marker}")

    return labels[str(top3[0])]


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clash Royale character classifier")
    parser.add_argument("--data_dir", type=str, help="Chemin vers image_loack")
    parser.add_argument("--predict",  type=str, help="Image a classifier")
    args = parser.parse_args()

    if args.predict:
        result = predict(args.predict)
        print(f"\n-> Personnage detecte : {result}")
    elif args.data_dir:
        train(args.data_dir)
    else:
        parser.print_help()
