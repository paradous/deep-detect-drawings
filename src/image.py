
import numpy as np
import torch
from skimage.transform import resize


class Image:

    def __init__(self, image: np.array):
        self.image = image

        self.streamlit = None  # Representation of what the computer see
        self.prediction = None  # Image array sanitized, to make a prediction

    def get_streamlit_displayable(self):
        """
        Return a representation of what the computer see. This image is still
        displayable in Streamlit.

        :return: A numpy array of the image.
        :rtype: np.array
        """

        if self.streamlit is None:

            # Grayscale the image
            gray_scaled = np.dot(self.image[..., :3], [0.2989, 0.5870, 0.1140])

            # Scale from [0, 255] to [0, 1]
            scaled = gray_scaled / 255

            # Resize to 28x28
            self.streamlit = resize(scaled, (28, 28))

        # Return a 28x28 (w, h) [0, 1] image
        return self.streamlit

    def get_prediction_ready(self):
        """
        Create a prediction-ready sanitized version of the image. This is what
        the model needs to make a prediction.

        :return: a torch Tensor of the image.
        :rtype: torch.Tensor
        """

        self.get_streamlit_displayable()

        if self.prediction is None:

            # Reshape to (1, 1, 28, 28)
            width, height, *_ = self.streamlit.shape
            reshaped = self.streamlit.reshape(1, 1, width, height)

            # Transform to float32
            img_float32 = reshaped.astype('float32')

            # Create a tensor
            self.prediction = torch.Tensor(img_float32)

        return self.prediction

    def is_empty(self):
        """Return true if self.image is an empty black image."""

        if np.max(self.streamlit) == np.min(self.streamlit) == .0:
            return True

        return False
