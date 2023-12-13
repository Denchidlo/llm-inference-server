from abc import ABC, abstractmethod
import typing as t
import math

class BaseLLM(ABC):
    @abstractmethod
    def infer_batch(self, batch: list[str], **infer_params) -> list[str]:
        raise NotImplementedError()
    

    def _split_to_mini_batches(self, data: list[str], mini_batch_size: int) -> t.Generator[list[str], t.Any, None]:
        for i in range(math.ceil(len(data) / mini_batch_size)):
            yield data[i * mini_batch_size:min((i + 1) * mini_batch_size, len(data))]