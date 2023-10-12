from .base import FacialTurnDetector
from .base import FacialTurnDetectorTorchModel
from .base import FacialTurnDetectorFeatureExtractor
import torch.nn as nn
import torch


class FacialTurnDetectorModel(FacialTurnDetectorTorchModel):
    def __init__(self, in_feature_dim):
        super().__init__()
        self._feature_dim = in_feature_dim

        self.linear_in1 = nn.Linear(self._feature_dim, 128)
        self.linear_in1b = nn.BatchNorm1d(128)
        self.linear_in2 = nn.Linear(128, 64)
        self.linear_in2b = nn.BatchNorm1d(64)
        self.lstm1 = nn.LSTM(64, 512, num_layers=3, dropout=0.5)
        self.linear_1 = nn.Linear(512, 128)
        self.linear_1b = nn.BatchNorm1d(128)
        self.linear_2 = nn.Linear(128, 64)
        self.linear_2b = nn.BatchNorm1d(64)
        self.linear_out_turn = nn.Linear(64, 2)  # 0 (OFF), 1 (ON)

        self._context = None

    def reset_context(self):
        self._context = None

    def detach_context(self):
        if self._context is None:
            return
        self._context = tuple([c.clone().detach() for c in self._context])

    def forward(self, feat):
        max_length, batch, feat_dim = feat.shape
        feat = feat.reshape(max_length * batch, -1)
        h_linear_in1 = torch.tanh(self.linear_in1(feat))
        # h_linear_in1 = self.linear_in1b(h_linear_in1)
        h_linear_in2 = torch.tanh(self.linear_in2(h_linear_in1))
        # h_linear_in2 = self.linear_in2b(h_linear_in2)
        h_linear_in2 = h_linear_in2.reshape(max_length, batch, -1)
        h_lstm1, context = self.lstm1(h_linear_in2, self._context)
        h_lstm1 = h_lstm1.reshape(max_length * batch, -1)
        h_lstm1 = torch.tanh(h_lstm1)
        h_linear1 = torch.tanh(self.linear_1(h_lstm1))
        # h_linear1 = self.linear_1b(h_linear1)
        h_linear2 = torch.tanh(self.linear_2(h_linear1))
        # h_linear2 = self.linear_2b(h_linear2)
        h_turn = self.linear_out_turn(h_linear2)
        h_turn = h_turn.reshape(max_length, batch, -1)

        self._context = context
        
        return h_turn


class FacialTurnDetector0001(FacialTurnDetector):
    def __init__(self,
                 feature_extractor: FacialTurnDetectorFeatureExtractor,
                 *args, **kwargs):
        super().__init__(feature_extractor, *args, **kwargs)
        self._model = FacialTurnDetectorModel(feature_extractor.feature_dim)
        if self.device:
            self._model = self._model.to(self.device)

    @property
    def torch_model(self) -> FacialTurnDetectorTorchModel:
        return self._model
        
    def convert_result_to_task_softmax(self,
                                       result: torch.Tensor) -> torch.Tensor:
        return torch.softmax(result, dim=2)
