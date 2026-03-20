import os
import platform
import settings
import phantomcv_helper as hp

class GCVWorker:
    def __init__(self, width, height):
        os.chdir(os.path.dirname(__file__))
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
