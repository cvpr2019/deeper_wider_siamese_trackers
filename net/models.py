from .siamfc import SiamFC_
from .features import ResNet23
from .connect import Corr_Up


class SiamFC_plus(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFC_plus, self).__init__(**kwargs)
        self.features = ResNet23()
        self.connect_model = Corr_Up()







