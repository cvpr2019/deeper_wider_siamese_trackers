import torch.nn as nn


class SiamFC_(nn.Module):
    def __init__(self):
        super(SiamFC_, self).__init__()
        self.features = None
        self.connect_model = None
        self.zf = None  # for online tracking

    def feature_extractor(self, x):
        return self.features(x)

    def connector(self, template_feature, search_feature):
        pred_score = self.connect_model(template_feature, search_feature)
        return pred_score

    def template(self, z):
         self.zf = self.feature_extractor(z)

    def track(self, x):
        xf = self.feature_extractor(x)
        pred_score = self.connector(self.zf, xf)
        return pred_score

