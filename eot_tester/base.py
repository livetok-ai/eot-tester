"""Base classes for EOT algorithms."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, Optional


class EOTState(Enum):
    """Possible end-of-turn states."""
    FINISHED = "finished"      # Complete thought, expecting response
    WAIT = "wait"             # User instructing AI not to speak
    UNFINISHED = "unfinished" # User paused but intends to continue


class EOTResult:
    """Result of end-of-turn detection."""
    
    def __init__(
        self, 
        state: EOTState, 
        confidence: float = 1.0, 
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.state = state
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"EOTResult(state={self.state.value}, confidence={self.confidence:.3f})"


class EOTAlgorithm(ABC):
    """Base class for all end-of-turn detection algorithms."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def detect(self, text: str, context: Optional[str] = None) -> EOTResult:
        """
        Detect end-of-turn state for given text.
        
        Args:
            text: The input text to analyze
            context: Optional context or system prompt
            
        Returns:
            EOTResult with detected state and confidence
        """
        pass
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the algorithm (load models, etc.)."""
        pass
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"