import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.pipeline import EvoPlanetPipeline
pipeline = EvoPlanetPipeline()
print("Model keys:")
for k in pipeline.detector.state_dict().keys():
    if "gate" in k:
        print(f"HAS GATE KEY: {k}")
        
try:
    pipeline.detector.load_state_dict(torch.load("weights/detector.pt", map_location='cpu'))
    print("LOADED SUCESSFULLY")
except Exception as e:
    print(f"FAILED TO LOAD: {e}")
