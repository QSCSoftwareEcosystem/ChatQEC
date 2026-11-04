"""Decoder base classes from decoder-bench's sampler.py, copied verbatim.

Source: https://github.com/satvikmaurya/decoder-bench, commit
9733a55a3a0e37c618294bd2ebbe0a8d4c334943 (MIT, see LICENSE). Only DecoderState, Decoder and
DecoderRegistry are kept: the HDF5 dataset sampler is not needed, so h5py is not required.
"""
# ruff: noqa
# mypy: ignore-errors
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Type

import numpy as np
import stim

@dataclass
class DecoderState:
    """Shared state for decoders that can be initialized once and reused"""
    check_matrix: Any
    obs_matrix: Any
    priors: np.ndarray
    circuit: stim.Circuit
    dem: stim.DetectorErrorModel

class Decoder(ABC):
    """Abstract base class for all decoders"""
    
    @abstractmethod
    def __init__(self, state: DecoderState, **kwargs):
        """Initialize the decoder with the given state and parameters"""
        pass
    
    @abstractmethod
    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """Decode a single syndrome and return a prediction"""
        pass
    
    @abstractmethod
    def verify(self, prediction: np.ndarray, observable: np.ndarray) -> bool:
        """Verify if the prediction matches the observable"""
        pass
    
    def decode_batch(self, syndromes: np.ndarray) -> List[np.ndarray]:
        """Decode a batch of syndromes (can be overridden for efficiency)"""
        return [self.decode(syndrome) for syndrome in syndromes]
    
    def verify_batch(self, predictions: List[np.ndarray], observables: np.ndarray) -> List[bool]:
        """Verify a batch of predictions (can be overridden for efficiency)"""
        return [self.verify(pred, obs) for pred, obs in zip(predictions, observables)]

class DecoderRegistry:
    """Registry for available decoders"""
    _decoders: Dict[str, Type[Decoder]] = {}
    
    @classmethod
    def register(cls, name: str = None):
        """Register a decoder class (can be used as a decorator)"""
        def decorator(decoder_class):
            decoder_name = name or decoder_class.__name__.lower()
            cls._decoders[decoder_name] = decoder_class
            return decoder_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[Decoder]:
        """Get a decoder class by name"""
        if name not in cls._decoders:
            raise ValueError(f"Decoder '{name}' not registered")
        return cls._decoders[name]
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available decoders"""
        return list(cls._decoders.keys())
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load decoder implementations from a Python file"""
        import importlib.util
        import sys
        
        module_name = os.path.basename(filepath).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        return len([d for d in cls._decoders.keys() if hasattr(module, d)])
    
    @classmethod
    def load_from_directory(cls, directory: str):
        """Load all decoder implementations from a directory"""
        loaded = 0
        for filename in os.listdir(directory):
            if filename.endswith('.py'):
                filepath = os.path.join(directory, filename)
                loaded += cls.load_from_file(filepath)
        return loaded
