#!/usr/bin/env python3
"""
Quick test script to verify Gemini 2.5 Pro connection
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_gemini_25():
    """Test Gemini 2.5 Pro connection"""

    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY or GOOGLE_API_KEY not set")
        print("\nPlease set your API key:")
        print("  export GEMINI_API_KEY='your-key-here'")
        return False

    print("✅ API key found")

    # Test import
    try:
        import google.generativeai as genai
        print("✅ google-generativeai library imported")
    except ImportError:
        print("❌ Error: google-generativeai not installed")
        print("\nPlease install it:")
        print("  pip install google-generativeai")
        return False

    # Configure API
    try:
        genai.configure(api_key=api_key)
        print("✅ API configured")
    except Exception as e:
        print(f"❌ Error configuring API: {e}")
        return False

    # Test Gemini 2.5 Pro
    model_name = "gemini-2.5-pro"
    print(f"\n🧪 Testing {model_name}...")

    try:
        model = genai.GenerativeModel(model_name)
        print(f"✅ Model '{model_name}' initialized")

        # Simple test prompt
        test_prompt = "What is 2+2? Answer with just the number."
        print(f"\nSending test prompt: '{test_prompt}'")

        response = model.generate_content(
            test_prompt,
            generation_config={
                'temperature': 0.1,
                'max_output_tokens': 50,
            }
        )

        print(f"✅ Response received: {response.text.strip()}")
        print(f"\n✨ SUCCESS! Gemini 2.5 Pro is working correctly!")
        return True

    except Exception as e:
        print(f"❌ Error testing model: {e}")

        # Try experimental version
        print(f"\n🧪 Trying experimental version: gemini-2.5-pro-exp-03-25")
        try:
            model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
            response = model.generate_content(test_prompt)
            print(f"✅ Experimental version works! Response: {response.text.strip()}")
            print(f"\n⚠️  Note: Use 'gemini-2.5-pro-exp-03-25' instead of 'gemini-2.5-pro'")
            return True
        except Exception as e2:
            print(f"❌ Experimental version also failed: {e2}")
            return False

if __name__ == "__main__":
    print("="*60)
    print("Gemini 2.5 Pro Connection Test")
    print("="*60)
    print()

    success = test_gemini_25()

    print()
    print("="*60)
    if success:
        print("✅ Test passed! You can now use Gemini 2.5 Pro")
        sys.exit(0)
    else:
        print("❌ Test failed - see errors above")
        sys.exit(1)
