"""Tests for TEN algorithm."""

import pytest
from unittest.mock import Mock, patch

from eot_tester.algorithms.ten import TENAlgorithm
from eot_tester.base import EOTState


@pytest.fixture
def ten_algorithm():
    """Create TEN algorithm instance."""
    return TENAlgorithm()


def test_ten_algorithm_creation(ten_algorithm):
    """Test TEN algorithm creation."""
    assert ten_algorithm.name == "TEN"
    assert ten_algorithm.model is None
    assert ten_algorithm.tokenizer is None


@patch('eot_tester.algorithms.ten.TRANSFORMERS_AVAILABLE', False)
def test_ten_algorithm_missing_dependencies(ten_algorithm):
    """Test TEN algorithm with missing dependencies."""
    with pytest.raises(ImportError, match="transformers and torch are required"):
        ten_algorithm.initialize()


@patch('eot_tester.algorithms.ten.TRANSFORMERS_AVAILABLE', True)
@patch('eot_tester.algorithms.ten.AutoModelForCausalLM')
@patch('eot_tester.algorithms.ten.AutoTokenizer')
@patch('eot_tester.algorithms.ten.torch')
def test_ten_algorithm_initialization(mock_torch, mock_tokenizer_cls, mock_model_cls, ten_algorithm):
    """Test TEN algorithm initialization."""
    # Mock torch.cuda.is_available()
    mock_torch.cuda.is_available.return_value = False
    mock_torch.bfloat16 = "bfloat16"
    
    # Mock model and tokenizer
    mock_model = Mock()
    mock_tokenizer = Mock()
    mock_tokenizer.eos_token = "<eos>"
    mock_tokenizer.pad_token = None
    
    mock_model_cls.from_pretrained.return_value = mock_model
    mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
    
    ten_algorithm.initialize()
    
    assert ten_algorithm.model == mock_model
    assert ten_algorithm.tokenizer == mock_tokenizer
    assert mock_tokenizer.pad_token == "<eos>"


def test_ten_algorithm_detect_not_initialized(ten_algorithm):
    """Test TEN algorithm detect without initialization."""
    with pytest.raises(RuntimeError, match="Model not initialized"):
        ten_algorithm.detect("test")


@patch('eot_tester.algorithms.ten.TRANSFORMERS_AVAILABLE', True)
@patch('eot_tester.algorithms.ten.torch')
def test_ten_algorithm_detect_empty_text(mock_torch, ten_algorithm):
    """Test TEN algorithm with empty text."""
    ten_algorithm.model = Mock()
    ten_algorithm.tokenizer = Mock()
    
    result = ten_algorithm.detect("")
    assert result.state == EOTState.UNFINISHED
    assert result.confidence == 0.5


@patch('eot_tester.algorithms.ten.TRANSFORMERS_AVAILABLE', True)
@patch('eot_tester.algorithms.ten.torch')
def test_ten_algorithm_parse_result(mock_torch, ten_algorithm):
    """Test TEN algorithm result parsing."""
    # Test exact matches
    assert ten_algorithm._parse_result("finished") == EOTState.FINISHED
    assert ten_algorithm._parse_result("wait") == EOTState.WAIT
    assert ten_algorithm._parse_result("unfinished") == EOTState.UNFINISHED
    
    # Test with additional text
    assert ten_algorithm._parse_result("the user is finished speaking") == EOTState.FINISHED
    assert ten_algorithm._parse_result("please wait") == EOTState.WAIT
    
    # Test fallback logic
    assert ten_algorithm._parse_result("hello world.") == EOTState.FINISHED
    assert ten_algorithm._parse_result("hello world") == EOTState.UNFINISHED


def test_ten_algorithm_calculate_confidence(ten_algorithm):
    """Test TEN algorithm confidence calculation."""
    # High confidence for exact matches
    assert ten_algorithm._calculate_confidence("finished") == 1.0
    assert ten_algorithm._calculate_confidence("wait") == 1.0
    assert ten_algorithm._calculate_confidence("unfinished") == 1.0
    
    # Medium confidence for keyword matches
    assert ten_algorithm._calculate_confidence("the user is finished") == 0.8
    assert ten_algorithm._calculate_confidence("please wait") == 0.8
    
    # Low confidence for no matches
    assert ten_algorithm._calculate_confidence("hello world") == 0.3