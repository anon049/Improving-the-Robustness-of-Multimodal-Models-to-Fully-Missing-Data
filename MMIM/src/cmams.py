from torch.nn import Module, Sequential, Linear, ReLU, Dropout, BatchNorm1d
from modules.encoders import RNNEncoder
import torch


class CMAM(Module):

    def __init__(self, hp, acoustic_enc_state_dict=None, visual_enc_state_dict=None):
        super(CMAM, self).__init__()
        self.visual_enc = RNNEncoder(
            in_size=hp.d_vin,
            hidden_size=hp.d_vh,
            out_size=hp.d_vout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_v if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional,
        )

        if visual_enc_state_dict is not None:
            self.visual_enc.load_state_dict(torch.load(visual_enc_state_dict))
            print("Visual encoder loaded")

        self.acoustic_enc = RNNEncoder(
            in_size=hp.d_ain,
            hidden_size=hp.d_ah,
            out_size=hp.d_aout,
            num_layers=hp.n_layer,
            dropout=hp.dropout_a if hp.n_layer > 1 else 0.0,
            bidirectional=hp.bidirectional,
        )

        if acoustic_enc_state_dict is not None:
            self.acoustic_enc.load_state_dict(torch.load(acoustic_enc_state_dict))
            print("Acoustic encoder loaded")

        ## 16 + 16
        ## 768

        self.fusion_model = Sequential(
            Linear(32, 64),
            BatchNorm1d(64),
            ReLU(),
            Dropout(0.4),
            Linear(64, 128),
            BatchNorm1d(128),
            ReLU(),
            Linear(128, 256),
            BatchNorm1d(256),
            ReLU(),
            Linear(256, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(0.6),
            Linear(512, 768),
        )

    def forward(self, audio, video):
        audio, audio_lengths = audio
        video, video_lengths = video
        audio_emb = self.acoustic_enc(audio, audio_lengths)
        video_emb = self.visual_enc(video, video_lengths)

        fused_emb = torch.cat((audio_emb, video_emb), dim=1)

        txt = self.fusion_model(fused_emb)

        return txt
