from abc import ABC, abstractmethod


class Model(ABC):
    @property
    @abstractmethod
    def model(self):
        pass

    @property
    @abstractmethod
    def optimizer(self):
        pass

    @property
    @abstractmethod
    def scheduler(self):
        pass

    @abstractmethod
    def unfreeze_last_block(self):
        pass

    @abstractmethod
    def unfreeze_penultimate_block(self):
        pass
