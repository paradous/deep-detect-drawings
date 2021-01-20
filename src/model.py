
import torch
import numpy as np
from src import CNN


class Model:

    def __init__(self):

        # Load the model from its Object and weight
        self.model = CNN()
        self.model.load_state_dict(torch.load(
            './models/quickdraw_cnn_model.pt',
            map_location=torch.device('cpu')
        ))

        # Freeze it - Set to inference mode
        self.model.eval()

        # Model prediction
        self.predicted_class = None
        self.probabilities = None

    def predict(self, image: torch.Tensor):

        if self.predicted_class is None:
            self.probabilities = self.model(image).detach().numpy()
            self.predicted_class = np.argmax(self.probabilities, axis=-1)[0]

        return self.predicted_class

    def get_probabilities(self):
        return self.probabilities

