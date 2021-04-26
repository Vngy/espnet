import copy
from typing import Optional
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
import echotorch
import echotorch.nn.reservoir as etnn
from echotorch.utils.matrix_generation import matrix_factory

from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs


class EchoFrontend(AbsFrontend):
    """ESN frontend structure for ASR.
    Instead of conventional: 
    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    
    we do:
    STFT -> ESN -> (Embed) 
    
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        n_fft: int = 512,
        win_length: int = None,
        hop_length: int = 128,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 80,
        res_out: int = 25,
        fmin: int = None,
        fmax: int = None,
        htk: bool = False,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
        apply_stft: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        # Deepcopy (In general, dict shouldn't be used as default arg)
        frontend_conf = copy.deepcopy(frontend_conf)

        if apply_stft:
            self.stft = Stft(
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                center=center,
                window=window,
                normalized=normalized,
                onesided=onesided,
            )
        else:
            self.stft = None
        self.apply_stft = apply_stft

        if frontend_conf is not None:
            self.frontend = Frontend(idim=n_fft // 2 + 1, **frontend_conf)
        else:
            self.frontend = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.res_out = res_out
        w_connectivity = 0.20
        win_connectivity = 0.50
        spectral_radius = 0.9
        leak_rate = 0.95
        input_scaling = 1.0
        num_layers = 1
        bias_scaling = 0.0
        deep_esn_type = 'IF'
        hidden_dim = res_out #from constructor arg
        output_layer = True
        output_dim = 50
        if output_layer:
            self.output_dim = output_dim
        else:
            self.output_dim = res_out*num_layers
        
        #Generate weight matrices
        w_generator = matrix_factory.get_generator(name='normal', spectral_radius=spectral_radius, connectivity=w_connectivity)
        win_generator = matrix_factory.get_generator(name='normal', connectivity=win_connectivity)
        wbias_generator = matrix_factory.get_generator(name='normal', scale=bias_scaling)

        self.esc = etnn.DeepESN(n_layers = num_layers,
                                                  input_dim = n_fft // 2 + 1,
                                                  hidden_dim = hidden_dim,
                                                  output_dim = output_dim,
                                                  leak_rate = leak_rate,
                                                  w_generator = w_generator,
                                                  win_generator = win_generator,
                                                  wbias_generator = wbias_generator,
                                                  input_scaling = input_scaling,
                                                  input_type = deep_esn_type,
                                                  create_output=output_layer)
    

    def output_size(self) -> int:
        return self.output_dim

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        if self.stft is not None:
            input_stft, feats_lens = self._compute_stft(input, input_lengths)
        else:
            input_stft = ComplexTensor(input[..., 0], input[..., 1])
            feats_lens = input_lengths
        # 2. [Option] Speech enhancement
        if self.frontend is not None:
            assert isinstance(input_stft, ComplexTensor), type(input_stft)
            # input_stft: (Batch, Length, [Channel], Freq)
            input_stft, _, mask = self.frontend(input_stft, feats_lens)

        # 3. [Multi channel case]: Select a channel
        if input_stft.dim() == 4:
            # h: (B, T, C, F) -> h: (B, T, F)
            if self.training:
                # Select 1ch randomly
                ch = np.random.randint(input_stft.size(2))
                input_stft = input_stft[:, :, ch, :]
            else:
                # Use the first channel
                input_stft = input_stft[:, :, 0, :]

        # 4. STFT -> Power spectrum
        # h: ComplexTensor(B, T, F) -> torch.Tensor(B, T, F)
        input_power = input_stft.real ** 2 + input_stft.imag ** 2
        
        # 5. STFT -> ESN States
        # input_power: (Batch, [Channel,] Length, Freq)
        #           -> input_feats: (Batch, Length, Out_Dim)
        #input_feats = torch.transpose(self.esc(input_power), 1, 2)
        input_feats = self.esc(input_power).to(self.device)
        self.esn_state = input_feats.transpose(1, 2).unsqueeze(1)
        print("input feats shape: ", input_feats.shape, "\tInput Feats length: ", len(feats_lens))
        
        return input_feats, feats_lens

    def get_esn_state(self):
        return self.esn_state


    def _compute_stft(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        input_stft, feats_lens = self.stft(input, input_lengths)

        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # Change torch.Tensor to ComplexTensor
        # input_stft: (..., F, 2) -> (..., F)
        input_stft = ComplexTensor(input_stft[..., 0], input_stft[..., 1])
        return input_stft, feats_lens
