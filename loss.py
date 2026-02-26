from torch import nn


class Loss(nn.Moudle):
    def __init__(self):
        super().__init__()
        self.location_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()

    def forward(self, predicts, targets):
        predic_location = predicts[:, 0:4]
        predict_classes = predicts[:, 4:8]
        target_location = targets[:, 0:4]
        target_classes = targets[:, 4:8]
        location_loss_value = self.location_loss(predic_location, target_location)
        class_loss_value = self.class_loss(predict_classes, target_classes)
        return location_loss_value + class_loss_value


