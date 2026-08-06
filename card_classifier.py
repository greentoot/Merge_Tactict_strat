"""
card_classifier.py
-------------------
Une seule classe pour charger un modèle Keras + ses labels et prédire.

Avant : Adb.py avait 3 blocs quasi identiques (_model/_labels, _model2/_labels2,
_model3/_labels3 + predict_character/predict_board/predict_board_3d) chargés
au niveau module — donc chargés dès le simple `import Adb`, même pour un
usage qui n'en a pas besoin (ex. tests, ou import depuis un autre script).
Maintenant : une classe réutilisable, instanciée à la demande dans
VisualMaking.__init__, donc les modèles ne se chargent que si on lance
réellement le bot.
"""

import json

import cv2
import numpy as np
import tensorflow as tf


class CardClassifier:
    def __init__(self, model_path: str, labels_path: str, img_h: int, img_w: int):
        self.model = tf.keras.models.load_model(model_path)
        with open(labels_path) as f:
            self.labels = json.load(f)
        self.img_h = img_h
        self.img_w = img_w

    def predict(self, img_bgr: np.ndarray) -> str:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (self.img_w, self.img_h))
        arr = np.expand_dims(img_resized.astype("float32") / 255.0, 0)
        preds = self.model.predict(arr, verbose=0)[0]
        return self.labels[str(int(np.argmax(preds)))]
