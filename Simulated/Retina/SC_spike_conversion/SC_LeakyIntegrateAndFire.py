from root_config import *
from Simulated.Retina.SC_spike_conversion.SC_Abstract import AbstractSpikeConversion
from Simulated.Retina.SC_spike_conversion import register_class

@register_class("LeakyIntegrateAndFire")
class LIFSpikeConversion(AbstractSpikeConversion):
    def __init__(self, params, device):
        super(LIFSpikeConversion, self).__init__(params, device)
        
        self.decay_rate = 0.9
        self.threshold = 0.5


    def forward(self, bipolar_signals):

        # Input: bipolar_signals: (BS, Timesteps, H, W)
        # Output: spikes: (BS, Timesteps, H, W)

        spikes = []

        current_potential = torch.zeros_like(bipolar_signals[:, 0])
        
        for t in range(bipolar_signals.shape[1]):
            current_potential = current_potential * self.decay_rate + bipolar_signals[:, t].abs()

            spike_flag = (current_potential > self.threshold)
            spikes.append(spike_flag.float())

            current_potential[spike_flag] = 0.0

        spikes = torch.stack(spikes, dim=1)

        return spikes
