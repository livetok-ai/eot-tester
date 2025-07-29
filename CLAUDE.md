# EOT Tester Project Context

This project is a command-line tool for testing different end-of-turn (EOT) detection algorithms, with a focus on conversational AI applications.

## Project Structure

```
eot-tester/
├── pyproject.toml          # Modern Python project configuration
├── README.md               # User documentation
├── CLAUDE.md              # This file - project context for Claude
└── eot_tester/            # Main package
    ├── __init__.py        # Package initialization
    ├── base.py            # Base classes and interfaces
    ├── cli.py             # Command-line interface
    └── algorithms/        # EOT algorithm implementations
        ├── __init__.py    # Algorithm package
        └── ten.py         # TEN algorithm implementation
```

## Key Components

### Base Classes (`base.py`)
- `EOTState`: Enum defining three possible states (FINISHED, WAIT, UNFINISHED)
- `EOTResult`: Data class for algorithm results with state, confidence, and metadata
- `EOTAlgorithm`: Abstract base class that all algorithms must implement

### CLI Interface (`cli.py`)
- Built with Click for modern CLI experience
- Rich library for beautiful terminal output
- Commands: `test`, `interactive`, `list-algorithms`
- Supports verbose logging and colored output

### TEN Algorithm (`algorithms/ten.py`)
- Implements the TEN Turn Detection algorithm from HuggingFace
- Uses Qwen2.5-7B transformer model
- Classifies text into finished/wait/unfinished states
- Handles model loading, tokenization, and inference

## Dependencies

### Core Dependencies
- `click>=8.0.0` - Modern CLI framework
- `transformers>=4.30.0` - HuggingFace transformers for TEN model
- `torch>=2.0.0` - PyTorch for model inference
- `numpy>=1.21.0` - Numerical operations
- `rich>=13.0.0` - Beautiful terminal output

### Development Dependencies
- `pytest>=7.0.0` - Testing framework
- `black>=23.0.0` - Code formatting
- `isort>=5.12.0` - Import sorting
- `flake8>=6.0.0` - Linting
- `mypy>=1.0.0` - Type checking

## Usage Patterns

The tool is designed for:
1. **Single text testing**: `eot-test test "Hello world"`
2. **Interactive mode**: `eot-test interactive`
3. **Algorithm comparison**: Easy to add new algorithms by implementing `EOTAlgorithm`

## Development Guidelines

- All algorithms must inherit from `EOTAlgorithm`
- Use proper error handling and logging
- Follow type hints throughout
- Rich terminal output for better UX
- Modular architecture for easy algorithm addition

## Important Commands

- Install: `uv sync` or `pip install -e .`
- Run tests: `pytest`
- Format code: `black . && isort .`
- Type check: `mypy eot_tester/`
- Run tool: `eot-test --help`