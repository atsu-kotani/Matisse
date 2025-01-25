from abc import ABC, abstractmethod


class ProgressLogger(ABC):
    def __init__(self, experiment_name, required_image_resolution):
        pass

    @abstractmethod
    def log_progress(self, params, retina, cortex, num_gradient_updates):
        pass
