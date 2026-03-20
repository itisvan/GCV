import importlib.util
import os
import platform
import sys
import settings

# Force-load phantomcv_helper from the .py source file so that any stale
# compiled .pyd (built from the old torch.hub-based code) cannot take priority.
_hp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phantomcv_helper.py")
_spec = importlib.util.spec_from_file_location("phantomcv_helper", _hp_path)
hp = importlib.util.module_from_spec(_spec)
sys.modules["phantomcv_helper"] = hp
_spec.loader.exec_module(hp)

class GCVWorker:
    def __init__(self, width, height):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        # Create an instance of the Phantom class
        self.phantom_instance = hp.Phantom()
        if settings.aimAssist:
            self.phantom_instance.load_model(settings.modelName)

    def __del__(self):
        del self

    def process(self, frame):
        gcvdata = bytearray()
        if settings.aimAssist:
            frame, gcvdata = self.phantom_instance.process(frame, gcvdata)
        return (frame, gcvdata)
