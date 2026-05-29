"""
Plant Disease Classifier - Command Line Inference
Usage:
    python inference.py <image_path>
    python inference.py test_leaf.jpg
"""

import tensorflow as tf
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import argparse


class PlantDiseasePredictor:
    def __init__(self, model_path="/media/ayush/Chocolate/CANDIES/Kaggle/PM3 final/Plant-Disease-Classifier/Model/plant_disease_model_final.keras"):
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        self.model = tf.keras.models.load_model(self.model_path, compile=False)
        print("✅ Model loaded successfully!")
        
        # Load class names
        class_path = Path("/media/ayush/Chocolate/CANDIES/Kaggle/PM3 final/Plant-Disease-Classifier/class_names.json")
        with open(class_path, 'r') as f:
            self.class_names = json.load(f)
        
        print(f"✅ Loaded {len(self.class_names)} classes\n")

    def predict(self, image_path: str, show_image: bool = True):
        """Predict disease from image"""
        try:
            img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            predictions = self.model.predict(img_array, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_idx] * 100)
            predicted_class = self.class_names[predicted_idx]
            
            print(f"🌿 Prediction : {predicted_class}")
            print(f"📊 Confidence: {confidence:.2f}%\n")
            
            if show_image:
                plt.figure(figsize=(8, 8))
                plt.imshow(img)
                plt.title(f"{predicted_class}\n({confidence:.2f}%)", fontsize=14, color='green')
                plt.axis('off')
                plt.show()
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None, None


# ========================= COMMAND LINE INTERFACE =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease Classifier")
    parser.add_argument("image", nargs="?", help="Path to the leaf image")
    parser.add_argument("--no-show", action="store_true", help="Don't show image")
    args = parser.parse_args()

    try:
        predictor = PlantDiseasePredictor()
        
        if args.image:
            predictor.predict(args.image, show_image=not args.no_show)
        else:
            # Interactive mode
            image_path = input("Enter path to leaf image: ").strip()
            if image_path:
                predictor.predict(image_path)
            else:
                print("No image path provided.")
                
    except Exception as e:
        print(f"Error: {e}")
        print("\nUsage: python inference.py <image_path>")