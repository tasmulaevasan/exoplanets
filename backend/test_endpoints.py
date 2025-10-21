"""
Quick test script for backend endpoints
"""
import sys
sys.path.insert(0, '.')

from config import get_available_models

print("Testing get_available_models()...")
try:
    models = get_available_models()
    print(f"[OK] Success! Got {len(models)} models:")
    for model in models:
        print(f"  - {model['type']}: {model['name']}")
        print(f"    Available: {model['available']}")
        print(f"    Training time: {model['training_time_estimate']}")
    print("\n[OK] All tests passed!")
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
