from abc import ABC, abstractmethod

from nd_datascience.machine_learning.model.application.sequence_to_sequence.kind.time_series_forcating.parameter.parameters import Parameters as TimeSeriesForcastingParameters
from nd_datascience.machine_learning.model.application.sequence_to_sequence.sequence_to_sequence import SequenceToSequence
from nd_datascience.machine_learning.model.model import Model
import numpy as np

class TimeSeries(SequenceToSequence, ABC):

    def __init__(self, parameters: TimeSeriesForcastingParameters):
        self._parameters = parameters

    @abstractmethod
    def get_predictions(self, input_sequence)->np.ndarray:
        pass