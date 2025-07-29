"""Tests for CLI functionality."""

from unittest.mock import Mock, patch
import pytest
from click.testing import CliRunner

from eot_tester.cli import main
from eot_tester.base import EOTState, EOTResult


@pytest.fixture
def runner():
    """Create click test runner."""
    return CliRunner()


def test_main_help(runner):
    """Test main command help."""
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert "EOT Tester" in result.output


def test_list_algorithms_command(runner):
    """Test list-algorithms command."""
    result = runner.invoke(main, ['list-algorithms'])
    assert result.exit_code == 0
    assert "TEN" in result.output
    assert "Available EOT Algorithms" in result.output


@patch('eot_tester.cli.TENAlgorithm')
def test_test_command_success(mock_algo_cls, runner):
    """Test successful test command."""
    # Mock algorithm
    mock_algo = Mock()
    mock_algo.name = "TEN"
    mock_algo_cls.return_value = mock_algo
    
    # Mock result
    mock_result = EOTResult(EOTState.FINISHED, confidence=0.95, metadata={"raw_output": "finished"})
    mock_algo.detect.return_value = mock_result
    
    result = runner.invoke(main, ['test', 'Hello world'])
    
    assert result.exit_code == 0
    assert mock_algo.initialize.called
    assert mock_algo.detect.called_with('Hello world', None)


@patch('eot_tester.cli.TENAlgorithm')
def test_test_command_with_context(mock_algo_cls, runner):
    """Test test command with context."""
    mock_algo = Mock()
    mock_algo.name = "TEN"
    mock_algo_cls.return_value = mock_algo
    
    mock_result = EOTResult(EOTState.WAIT, confidence=0.8)
    mock_algo.detect.return_value = mock_result
    
    result = runner.invoke(main, ['test', 'Hold on', '--context', 'You are helpful'])
    
    assert result.exit_code == 0
    mock_algo.detect.assert_called_with('Hold on', 'You are helpful')


def test_test_command_unknown_algorithm(runner):
    """Test test command with unknown algorithm."""
    result = runner.invoke(main, ['test', 'Hello', '--algorithm', 'unknown'])
    
    assert result.exit_code == 0
    assert "Unknown algorithm: unknown" in result.output


@patch('eot_tester.cli.TENAlgorithm')
def test_test_command_initialization_error(mock_algo_cls, runner):
    """Test test command with initialization error."""
    mock_algo = Mock()
    mock_algo.initialize.side_effect = Exception("Model not found")
    mock_algo_cls.return_value = mock_algo
    
    result = runner.invoke(main, ['test', 'Hello'])
    
    assert result.exit_code == 0
    assert "Error:" in result.output


@patch('eot_tester.cli.TENAlgorithm')
@patch('eot_tester.cli.click.prompt')
def test_interactive_mode_quit(mock_prompt, mock_algo_cls, runner):
    """Test interactive mode with quit command."""
    mock_prompt.return_value = 'quit'
    
    mock_algo = Mock()
    mock_algo.name = "TEN"
    mock_algo_cls.return_value = mock_algo
    
    result = runner.invoke(main, ['interactive'])
    
    assert result.exit_code == 0
    assert mock_algo.initialize.called


@patch('eot_tester.cli.TENAlgorithm')
@patch('eot_tester.cli.click.prompt')
def test_interactive_mode_with_input(mock_prompt, mock_algo_cls, runner):
    """Test interactive mode with text input."""
    mock_prompt.side_effect = ['Hello world', 'quit']
    
    mock_algo = Mock()
    mock_algo.name = "TEN"
    mock_result = EOTResult(EOTState.FINISHED, confidence=0.9)
    mock_algo.detect.return_value = mock_result
    mock_algo_cls.return_value = mock_algo
    
    result = runner.invoke(main, ['interactive'])
    
    assert result.exit_code == 0
    assert mock_algo.detect.called