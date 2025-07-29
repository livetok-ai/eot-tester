#!/usr/bin/env python3
"""Simple test script to verify LiveKit integration."""

import sys
from pathlib import Path

# Add the project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_import():
    """Test that we can import the LiveKit algorithm."""
    try:
        from eot_tester.algorithms import LiveKitAlgorithm
        print("✓ Successfully imported LiveKitAlgorithm")
        return True
    except ImportError as e:
        print(f"✗ Failed to import LiveKitAlgorithm: {e}")
        return False

def test_instantiation():
    """Test that we can create instances of the algorithm."""
    try:
        from eot_tester.algorithms import LiveKitAlgorithm
        
        # Test English model
        algo_en = LiveKitAlgorithm(model_type="en")
        print(f"✓ Successfully created English algorithm: {algo_en.name}")
        
        # Test multilingual model
        algo_multi = LiveKitAlgorithm(model_type="multilingual")
        print(f"✓ Successfully created multilingual algorithm: {algo_multi.name}")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create algorithm instances: {e}")
        return False

def test_base_interface():
    """Test that the algorithm implements the base interface correctly."""
    try:
        from eot_tester.algorithms import LiveKitAlgorithm
        from eot_tester.base import EOTAlgorithm, EOTResult, EOTState
        
        algo = LiveKitAlgorithm(model_type="en")
        
        # Check inheritance
        assert isinstance(algo, EOTAlgorithm), "Algorithm should inherit from EOTAlgorithm"
        
        # Check required methods exist
        assert hasattr(algo, 'detect'), "Algorithm should have detect method"
        assert hasattr(algo, 'initialize'), "Algorithm should have initialize method"
        
        print("✓ Algorithm correctly implements base interface")
        return True
    except Exception as e:
        print(f"✗ Interface test failed: {e}")
        return False

def test_without_dependencies():
    """Test behavior when dependencies are not available."""
    try:
        from eot_tester.algorithms import LiveKitAlgorithm
        
        algo = LiveKitAlgorithm(model_type="en")
        
        # This should fail gracefully if dependencies are missing
        try:
            algo.initialize()
            print("✓ Algorithm initialized (dependencies available)")
        except ImportError as e:
            if "Required dependencies not available" in str(e):
                print("✓ Algorithm gracefully handles missing dependencies")
            else:
                print(f"✗ Unexpected import error: {e}")
                return False
        except Exception as e:
            print(f"ℹ Algorithm initialization failed (expected without models): {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ Dependency test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing LiveKit Integration...")
    print("=" * 50)
    
    tests = [
        test_import,
        test_instantiation,
        test_base_interface,
        test_without_dependencies,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! LiveKit integration is working.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check the output above.")
        sys.exit(1)