from abc import ABCMeta
from abc import abstractmethod

class Enzyme(metaclass=ABCMeta):
    inhibition_rate: float
    inhibited: bool

    @abstractmethod
    def inhibit(self, substance, dose_ug):
        pass

class CYP3A4(Enzyme):
    def __init__(self):
        self.inhibition_rate = 0
        self.inhibited = False
    
    def inhibit(self, substance, dose_ug):
        pass