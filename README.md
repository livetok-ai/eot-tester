# EOT Tester

A Python command-line tool for testing different end-of-turn (EOT) detection algorithms in conversational AI applications. 

It includes support for:

- **TEN Framework**
- **LiveKit EOU Model**
- **Pipecat Smart Turn v2**

This is an initial version of the testing tool with the following limitations:
- The testing dataset is currently from the TEN framework, so that implementation may have an advantage in evaluation results
- Only runs tests with English sentences
- For speech models like Pipecat Smart Turn v2, we artificially generate speech from texts in the dataset. Ideally, we would use audio for all models and perform STT before processing ones that require text input
- The TEN implementation runs slowly on macOS

## Installation

We recommend using [uv](https://docs.astral.sh/uv/) for fast Python package management:

```bash
# Install dependencies and the project
uv sync
```

Alternatively, you can use pip:

```bash
pip install -e .
```

## Usage

For help with available commands and options:

```bash
eot-test --help
```

### Basic Testing

Test a single piece of text:

```bash
eot-test test "I think that's all I wanted to say." --algorith=xxx
```

### Evaluation Mode

Start an interactive session for testing multiple inputs:

```bash
eot-test evaluate --algorith=xxx
```

### Verbose Output

Enable detailed logging:

```bash
eot-test --verbose test "Hello world"
```

## EOT States

The tool classifies text into three possible end-of-turn states:

- **FINISHED**: Complete thought expecting a response
- **UNFINISHED**: User momentarily paused but intends to continue

## Algorithms

The EOT Tester supports multiple state-of-the-art algorithms for end-of-turn detection:

### TEN Algorithm

The TEN (Turn Detection) algorithm uses a fine-tuned transformer-based model (Qwen2.5-7B) from the HuggingFace TEN-framework. It classifies text into three precise states:

- **finished**: Complete thought expecting a response
- **wait**: User instructing AI not to speak  
- **unfinished**: User momentarily paused but intends to continue


### LiveKit Algorithm

The LiveKit algorithm uses ONNX-optimized models from the `livekit/turn-detector` HuggingFace repository. It provides:

- **English-only model** (`v1.2.2-en`): Optimized for English conversations
- **Multilingual model** (`v0.3.0-intl`): Supports multiple languages with language-specific thresholds

### Pipecat Smart Turn V2 Algorithm

The Pipecat algorithm uses audio-based turn detection with a custom Wav2Vec2 model. Key features:

- **Audio-based detection**: Analyzes speech patterns rather than text
- **Text-to-speech conversion**: Converts input text to speech using OpenAI TTS API (`tts-1` model with `alloy` voice)

**States**: Binary classification (Complete/Incomplete) mapped to FINISHED/UNFINISHED.

## Development

### Setup Development Environment

```bash
# Install with development dependencies
uv sync --group dev

# Or with pip
pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code
black .
isort .

# Run linting
flake8 eot_tester/

# Type checking
mypy eot_tester/

# Run tests
pytest
```

### Adding New Algorithms

1. Create a new file in `eot_tester/algorithms/`
2. Inherit from `EOTAlgorithm` base class
3. Implement the required `detect()` and `initialize()` methods
4. Add your algorithm to the CLI in `cli.py`

Example:

```python
from ..base import EOTAlgorithm, EOTResult, EOTState

class MyAlgorithm(EOTAlgorithm):
    def __init__(self):
        super().__init__("MyAlgorithm")
    
    def initialize(self) -> None:
        # Load models, setup resources
        pass
    
    def detect(self, text: str, context: str = None) -> EOTResult:
        # Your detection logic here
        return EOTResult(EOTState.FINISHED, confidence=0.9)
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- HuggingFace Transformers 4.30+

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run the code quality checks
6. Submit a pull request