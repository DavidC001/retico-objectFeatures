import sys
sys.path.append("path/to/retico-vision")

from retico_core import *
from retico_core.debug import DebugModule
from retico_vision.vision import WebcamModule 
from retico_vision.vision import ExtractObjectsModule
from retico_yolov11.yolov11 import Yolov11
from retico_objectFeatures.objects_feat_extr import ObjectFeaturesExtractor

webcam = WebcamModule()
yolo = Yolov11()  
extractor = ExtractObjectsModule(num_obj_to_display=5)  
feats = ObjectFeaturesExtractor(show=True,top_objects=5)
debug = DebugModule()  

webcam.subscribe(yolo)  
yolo.subscribe(extractor)   
#sam.subscribe(debug)
extractor.subscribe(feats)    
feats.subscribe(debug)

webcam.run()  
yolo.run()  
extractor.run()  
feats.run()
debug.run()  

print("Network is running")
input()

webcam.stop()  
yolo.stop()  
extractor.stop()   
debug.stop()  