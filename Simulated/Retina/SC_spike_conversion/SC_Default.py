import torch
import numpy as np
from root_config import *
from Simulated.Retina.SC_spike_conversion.SC_Abstract import AbstractSpikeConversion
from Simulated.Retina.SC_spike_conversion import register_class


@register_class("Default")
class DefaultSpikeConversion(AbstractSpikeConversion):
    def __init__(self, params, device):
        super(DefaultSpikeConversion, self).__init__(params, device)

    def forward(self, bipolar_signals):
        return bipolar_signals
