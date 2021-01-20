
from torch import nn
from torch.nn import functional


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # Features detector
        self.features1 = nn.Sequential(

            # Hidden layer 1
            nn.Conv2d(1, 32, kernel_size=3), nn.ReLU(),
            nn.BatchNorm2d(32),

            # Hidden layer 2
            nn.Conv2d(32, 32, kernel_size=3), nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.2)
        )

        # Features detector
        self.features2 = nn.Sequential(

            # Hidden layer 3
            nn.Conv2d(32, 64, kernel_size=3), nn.ReLU(),
            nn.BatchNorm2d(64),

            # Hidden layer 4
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.3)
        )

        # Features detector
        self.features3 = nn.Sequential(

            # Hidden layer 4
            nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1)), nn.ReLU(),
            nn.BatchNorm2d(128),

            # Hidden layer 5
            nn.Conv2d(128, 128, kernel_size=3), nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.4),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),

            # Output layer
            nn.Linear(128, 2)
        )

    def forward(self, x):

        # Features
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)

        # Classifier
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return functional.log_softmax(x, dim=1)
