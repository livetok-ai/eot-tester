#!/usr/bin/env python3
"""Simple test script for Pipecat algorithm."""

import os
from eot_tester.algorithms.pipecat import PipecatAlgorithm

def test_pipecat():
    """Test the Pipecat algorithm with sample text."""
    
    # Check if OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("Warning: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your_api_key_here'")
        return
    
    # Initialize algorithm
    algo = PipecatAlgorithm()
    
    try:
        print("Initializing Pipecat algorithm...")
        algo.initialize()
        print("✓ Algorithm initialized successfully")
        
        # Test with sample text
        test_text = "Hello, how are you doing today?"
        print(f"Testing with text: '{test_text}'")
        
        result = algo.detect(test_text)
        
        print(f"Result: {result}")
        print(f"State: {result.state.value}")
        print(f"Confidence: {result.confidence:.3f}")
        
        if result.metadata:
            print("Metadata:")
            for key, value in result.metadata.items():
                if key == 'audio_path':
                    print(f"  {key}: {value}")
                elif key == 'prediction':
                    print(f"  {key}: {value} ({'Complete' if value == 1 else 'Incomplete'})")
                elif key == 'probability':
                    print(f"  {key}: {value:.3f}")
                elif key == 'device':
                    print(f"  {key}: {value}")
                elif key == 'raw_results':
                    print(f"  {key}: {value}")
                elif key == 'audio_length':
                    print(f"  {key}: {value:.2f}s")
                else:
                    print(f"  {key}: {value}")
        
        print("✓ Test completed successfully")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("Make sure you have:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Installed all dependencies with: uv sync")

if __name__ == "__main__":
    test_pipecat()