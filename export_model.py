"""
export_model.py
Minimal standalone script to export YOLO26 model weights to different formats.
No GUI, no webcam, just model conversion.
"""

from ultralytics import YOLO
import sys
import os

def export_yolo_model(model_path='yolo26n.pt'):
    if not os.path.exists(model_path):
        print(f"❌ Error: Model weights file '{model_path}' not found.")
        print(f"Current directory: {os.getcwd()}")
        return

    print(f"Loading YOLO26 model from {model_path}...")
    model = YOLO(model_path)

    # List of common export formats for graduation projects/deployment
    # format='onnx' -> Standard for most backends
    # format='tflite' -> For mobile/embedded
    # format='torchscript' -> For C++ deployment
    
    formats = ['onnx', 'tflite', 'torchscript']
    
    print("\nStarting model export journey...")
    print("="*40)
    
    for fmt in formats:
        try:
            print(f"\n📦 Exporting to {fmt.upper()}...")
            # Note: imgsz=320 is often best for speed on edge devices
            path = model.export(format=fmt, imgsz=320, verbose=False)
            print(f"✅ Success! Exported to: {path}")
        except Exception as e:
            print(f"❌ Failed to export {fmt}: {e}")

    print("\n" + "="*40)
    print("Export process completed!")

if __name__ == "__main__":
    export_yolo_model()
