"""
models_cnn.py -- Classifier 3: a small convolutional network for 32x32 ink images.

Why a CNN here: the MLP treats pixel (5,5) and pixel (5,6) as unrelated inputs, so it
has no idea what "next to" means and must memorise every position a stroke can be in.
A convolution slides ONE small filter over the whole image, computing the same weighted
sum at every location.  It learns "diagonal stroke" once and finds it anywhere.
That gives (a) far fewer weights and (b) tolerance to where the shape sits.

Shapes through the network (batch dimension omitted):
    input        1 x 32 x 32      the ink image
    conv1        16 x 32 x 32     16 filters, each 3x3, padding=1 keeps the size
    pool         16 x 16 x 16     max over each 2x2 block: halve the size, keep the strongest response
    conv2        32 x 16 x 16     32 filters, each looks at a 3x3 patch of ALL 16 channels below
    pool         32 x 8 x 8       now one cell ~ a 4x4 patch of the original image
    flatten      2048
    dropout      2048             (training only) randomly zero 30% so no unit is relied on
    fc1          64               one perceptron per unit over the 2048 features
    fc2          1                the logit: >0 X, <0 O
"""
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, size: int = 32, channels=(16, 32), fc: int = 64, dropout: float = 0.3):
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        side = size // 4                               # two poolings: 32 -> 16 -> 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c2 * side * side, fc), nn.ReLU(),
            nn.Linear(fc, 1),
        )

    def forward(self, x):                              # x: [batch, 1, size, size]
        return self.classifier(self.features(x)).squeeze(1)
