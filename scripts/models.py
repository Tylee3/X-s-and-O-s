"""
models.py -- the three classifiers, all as nn.Module so all three can be exported to the
class playground (input float32 [1,1,64,64] in [-1,1], output float32 [1,2] logits, O then X).

BackgroundNormalize  shared front block: Step-1's "darker than the local background" idea,
                     written in tensor ops so it is baked into the exported model.
ManualPerceptron     Classifier 1: fixed region features + hand-set weights, NO training.
MLP                  Classifier 2: flatten -> dense layers.
CNN                  Classifier 3: in models_cnn.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 64


class BackgroundNormalize(nn.Module):
    """[B,1,64,64] in [-1,1] (ink dark)  ->  [B,1,64,64] in [0,1] (ink bright, background 0).

    gray  = (x + 1) / 2                     back to 0..1, white = 1
    bg    = erode(dilate(gray))             morphological CLOSING: dilate = max over a window
                                            (fills the thin dark strokes with the bright surface),
                                            erode = min over a window (shrinks it back).  Result:
                                            what the surface would look like with nothing drawn.
    dark  = max(bg - gray, 0)               how much darker than its own surroundings each pixel is
    out   = dark / max(dark)                strongest stroke = 1, whatever the exposure (with a
                                            floor so a blank image stays ~0 instead of amplifying noise)
    No learnable parameters.  Every op (MaxPool, ReduceMax, Clip, Div) is a standard ONNX op.
    """

    def __init__(self, kernel: int = 11, floor: float = 10 / 255):
        super().__init__()
        self.k, self.floor = kernel, floor

    def forward(self, x):
        gray = (x + 1) * 0.5
        p = self.k // 2
        dilated = F.max_pool2d(gray, self.k, stride=1, padding=p)
        bg = -F.max_pool2d(-dilated, self.k, stride=1, padding=p)
        dark = torch.clamp(bg - gray, min=0.0)
        peak = torch.clamp(torch.amax(dark, dim=(2, 3), keepdim=True), min=self.floor)
        return torch.clamp(dark / peak, 0.0, 1.0)


# ----------------------------------------------------------------------------- #
# Classifier 1
# ----------------------------------------------------------------------------- #
def region_masks(n: int = SIZE) -> dict:
    """The four hand-designed regions (see features.py for the reasoning)."""
    yy, xx = torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")
    c = (n - 1) / 2
    r = torch.hypot(yy - c, xx - c)
    return {
        "centre": (torch.abs(yy - c) <= n / 8) & (torch.abs(xx - c) <= n / 8),      # 16x16 middle square
        "ring": (r >= 0.28 * n) & (r <= 0.44 * n),                                   # radius 18..28
        "diag": (torch.abs(yy - xx) <= n / 16) | (torch.abs(yy + xx - (n - 1)) <= n / 16),
        "corners": ((yy + xx) < 0.34 * n) | ((yy + xx) > 2 * (n - 1) - 0.34 * n)
                   | ((yy - xx) > 0.62 * n) | ((xx - yy) > 0.62 * n),
    }


FEATURE_NAMES = ["centre", "ring", "diag", "corners"]


class ManualPerceptron(nn.Module):
    """score = w . features + b ;  logits = [0, score]  so softmax gives P(X) = sigmoid(score).

    features[k] = mean ink in region k / mean ink in the whole image  (stroke-width independent).
    The weights are SET BY HAND in perceptron_manual.py and never trained: there is no
    optimizer, no loss, no backward pass anywhere near this class.
    """

    def __init__(self, weights=(1.0, -1.0, 0.0, 0.0), bias=-0.3):
        super().__init__()
        self.norm = BackgroundNormalize()
        masks = region_masks()
        self.register_buffer("masks", torch.stack([masks[k] for k in FEATURE_NAMES]).float())  # [4,64,64]
        self.register_buffer("w", torch.tensor(weights, dtype=torch.float32))
        self.register_buffer("b", torch.tensor(float(bias)))

    def features(self, x):
        ink = self.norm(x)                                               # [B,1,64,64]
        avg = ink.mean(dim=(1, 2, 3), keepdim=True) + 1e-6               # [B,1,1,1]
        region_sum = (ink * self.masks.unsqueeze(0)).sum(dim=(2, 3))     # [B,4]
        region_mean = region_sum / self.masks.sum(dim=(1, 2)).unsqueeze(0)
        return region_mean / avg.squeeze(-1).squeeze(-1)                 # [B,4] ratios

    def forward(self, x):
        score = self.features(x) @ self.w + self.b                       # [B]
        return torch.stack([torch.zeros_like(score), score], dim=1)      # [B,2] logits (O, X)


# ----------------------------------------------------------------------------- #
# Classifier 2
# ----------------------------------------------------------------------------- #
class MLP(nn.Module):
    """BackgroundNormalize -> flatten 4096 -> hidden[0] -> hidden[1] -> 2 logits.

    Linear  : every unit = weighted sum of ALL 4096 inputs + bias (one perceptron each).
    ReLU    : max(0, x).  Without it, stacked linear layers multiply into ONE linear layer,
              i.e. the single straight line that could not separate the flat ovals in Step 3.
    Dropout : training only; randomly zero 30% of units so none can be relied on.
    Output  : 2 logits.  With two classes this is equivalent to one logit + sigmoid; the
              playground contract requires the 2-logit form, ordered (O, X).
    """

    def __init__(self, size: int = SIZE, hidden=(128, 64), dropout: float = 0.3):
        super().__init__()
        self.norm = BackgroundNormalize()
        layers, n_in = [], size * size
        for n_out in hidden:
            layers += [nn.Linear(n_in, n_out), nn.ReLU(), nn.Dropout(dropout)]
            n_in = n_out
        layers.append(nn.Linear(n_in, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(self.norm(x).flatten(1))


def build(name: str, **kw) -> nn.Module:
    if name == "mlp":
        return MLP(**kw)
    if name == "cnn":
        from models_cnn import CNN
        return CNN(**kw)
    if name == "perceptron":
        return ManualPerceptron(**kw)
    raise ValueError(name)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
