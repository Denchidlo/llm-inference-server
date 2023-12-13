from abc import ABC, abstractmethod

class BaseLLM(ABC):
    @abstractmethod
    def infer_batch(self, batch: list[str], **infer_params) -> list[str]:
        raise NotImplementedError()
