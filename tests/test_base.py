"""Tests for base classes."""

import pytest

from eot_tester.base import EOTState, EOTResult, EOTAlgorithm


def test_eot_state_enum():
    """Test EOTState enum values."""
    assert EOTState.FINISHED.value == "finished"
    assert EOTState.WAIT.value == "wait"
    assert EOTState.UNFINISHED.value == "unfinished"


def test_eot_result_creation():
    """Test EOTResult creation and string representation."""
    result = EOTResult(EOTState.FINISHED, confidence=0.95)
    
    assert result.state == EOTState.FINISHED
    assert result.confidence == 0.95
    assert result.metadata == {}
    assert "finished" in str(result)
    assert "0.950" in str(result)


def test_eot_result_with_metadata():
    """Test EOTResult with metadata."""
    metadata = {"raw_output": "finished", "model": "test"}
    result = EOTResult(EOTState.FINISHED, metadata=metadata)
    
    assert result.metadata["raw_output"] == "finished"
    assert result.metadata["model"] == "test"


class MockAlgorithm(EOTAlgorithm):
    """Mock algorithm for testing."""
    
    def __init__(self):
        super().__init__("MockAlgorithm")
        self.initialized = False
    
    def initialize(self):
        self.initialized = True
    
    def detect(self, text, context=None):
        if not self.initialized:
            raise RuntimeError("Not initialized")
        return EOTResult(EOTState.FINISHED, confidence=1.0)


def test_algorithm_base_class():
    """Test EOTAlgorithm base class functionality."""
    algo = MockAlgorithm()
    
    assert algo.name == "MockAlgorithm"
    assert "MockAlgorithm" in str(algo)
    
    # Should raise error before initialization
    with pytest.raises(RuntimeError):
        algo.detect("test")
    
    # Should work after initialization
    algo.initialize()
    result = algo.detect("test")
    
    assert result.state == EOTState.FINISHED
    assert result.confidence == 1.0