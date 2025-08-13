#!/usr/bin/env python3
# Copyright         2024  Xiaomi Corp.        (authors: Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torchaudio
from lhotse.features.base import FeatureExtractor, register_extractor
from lhotse.utils import Seconds, compute_num_frames
from zipvoice.utils._bigvgan_mel_feature import mel_spectrogram as bigvgan_mel_spectrogram

@dataclass
class VocosFbankConfig:
    sampling_rate: int = 24000
    n_mels: int = 100
    n_fft: int = 1024
    hop_length: int = 256


@register_extractor
class VocosFbank(FeatureExtractor):

    name = "VocosFbank"
    config_type = VocosFbankConfig

    def __init__(self, num_channels: int = 1):
        config = VocosFbankConfig
        super().__init__(config=config)
        assert num_channels in (1, 2)
        self.num_channels = num_channels
        self.fbank = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sampling_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            center=True,
            power=1,
        )

    def _feature_fn(self, sample):
        mel = self.fbank(sample)
        logmel = mel.clamp(min=1e-7).log()

        return logmel

    @property
    def device(self) -> Union[str, torch.device]:
        return self.config.device

    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.n_mels

    def extract(
        self,
        samples: Union[np.ndarray, torch.Tensor],
        sampling_rate: int,
    ) -> Union[np.ndarray, torch.Tensor]:
        # Check for sampling rate compatibility.
        expected_sr = self.config.sampling_rate
        assert sampling_rate == expected_sr, (
            f"Mismatched sampling rate: extractor expects {expected_sr}, "
            f"got {sampling_rate}"
        )
        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        else:
            assert samples.ndim == 2, samples.shape

        if self.num_channels == 1:
            if samples.shape[0] == 2:
                samples = samples.mean(dim=0, keepdims=True)
        else:
            assert samples.shape[0] == 2, samples.shape

        mel = self._feature_fn(samples)
        # (1, n_mels, time) or (2, n_mels, time)
        mel = mel.reshape(-1, mel.shape[-1]).t()
        # (time, n_mels) or (time, 2 * n_mels)

        num_frames = compute_num_frames(
            samples.shape[1] / sampling_rate, self.frame_shift, sampling_rate
        )

        if mel.shape[0] > num_frames:
            mel = mel[:num_frames]
        elif mel.shape[0] < num_frames:
            mel = mel.unsqueeze(0)
            mel = torch.nn.functional.pad(
                mel, (0, 0, 0, num_frames - mel.shape[1]), mode="replicate"
            ).squeeze(0)

        if is_numpy:
            return mel.cpu().numpy()
        else:
            return mel

    @property
    def frame_shift(self) -> Seconds:
        return self.config.hop_length / self.config.sampling_rate

@dataclass
class BigVGANFbankConfig:
    n_fft: int = 1024
    num_mels: int = 100
    sampling_rate: int = 24000
    hop_size: int = 256
    win_size: int = 1024
    fmin: int = 0
    fmax: int = None # 12000 for bigvgan v1.
    
    
@register_extractor
class BigVGANFbank(FeatureExtractor):
    config_type = BigVGANFbankConfig
    name = "BigVGANFbank"
    def __init__(self, num_channels: int = 1):
        config = BigVGANFbankConfig
        super().__init__(config=config)
        assert num_channels in (1, 2)
        self.num_channels = num_channels
        
    def feature_dim(self, sampling_rate: int) -> int:
        return self.config.num_mels * self.num_channels
    
    @property
    def frame_shift(self) -> Seconds:
        return self.config.hop_size / self.config.sampling_rate
        
    def extract(self, samples, sampling_rate):
        # samples: (channel, time) or (time,)
        # Check for sampling rate compatibility.
        expected_sr = self.config.sampling_rate
        assert sampling_rate == expected_sr, (
            f"Mismatched sampling rate: extractor expects {expected_sr}, "
            f"got {sampling_rate}"
        )
        is_numpy = False
        if not isinstance(samples, torch.Tensor):
            samples = torch.from_numpy(samples)
            is_numpy = True

        if len(samples.shape) == 1:
            samples = samples.unsqueeze(0)
        else:
            assert samples.ndim == 2, samples.shape

        if self.num_channels == 1:
            if samples.shape[0] == 2:
                samples = samples.mean(dim=0, keepdims=True)
        else:
            assert samples.shape[0] == 2, samples.shape

        mel = bigvgan_mel_spectrogram(
            samples,
            n_fft=self.config.n_fft,
            num_mels=self.config.num_mels,
            sampling_rate=self.config.sampling_rate,
            hop_size=self.config.hop_size,
            win_size=self.config.win_size,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
            center=False,
        )
        # (1, n_mels, time) or (2, n_mels, time)
        mel = mel.reshape(-1, mel.shape[-1]).t()
        # (time, n_mels) or (time, 2 * n_mels)

        num_frames = compute_num_frames(
            samples.shape[1] / sampling_rate, self.frame_shift, sampling_rate
        )

        if mel.shape[0] > num_frames:
            mel = mel[:num_frames]
        elif mel.shape[0] < num_frames:
            mel = mel.unsqueeze(0)
            mel = torch.nn.functional.pad(
                mel, (0, 0, 0, num_frames - mel.shape[1]), mode="replicate"
            ).squeeze(0)

        if is_numpy:
            return mel.cpu().numpy()
        else:
            return mel