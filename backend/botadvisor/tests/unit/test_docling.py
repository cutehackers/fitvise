#!/usr/bin/env python3
"""
Test script to understand Docling API
"""

import sys
sys.path.append('.')

try:
    import docling
    print("✅ Docling imported successfully")
    print(f"Docling version: {getattr(docling, '__version__', 'unknown')}")

    # Try to explore the API
    print("\nAvailable attributes in docling:")
    for attr in dir(docling):
        if not attr.startswith('_'):
            print(f"  - {attr}")

    # Try to find readers
    if hasattr(docling, 'readers'):
        print(f"\n✅ docling.readers found")
        readers_module = docling.readers
        print(f"Available in docling.readers:")
        for attr in dir(readers_module):
            if not attr.startswith('_'):
                print(f"  - {attr}")
    else:
        print(f"\n❌ docling.readers not found")

    # Try to find PDF reading functionality
    print(f"\nLooking for PDF reading functionality...")
    if hasattr(docling, 'PDFReader'):
        print(f"✅ docling.PDFReader found")
    elif hasattr(docling, 'read_pdf'):
        print(f"✅ docling.read_pdf found")
    else:
        print(f"❌ No obvious PDF reading function found")

    # Try to read a simple text file
    print(f"\nTrying to read text file...")
    try:
        with open('test_sample.txt', 'r') as f:
            content = f.read()
        print(f"File content length: {len(content)} characters")

        # Try different approaches to read text
        approaches = [
            ('docling.TextReader', lambda: hasattr(docling, 'TextReader')),
            ('docling.read_text', lambda: hasattr(docling, 'read_text')),
            ('docling.read', lambda: hasattr(docling, 'read')),
        ]

        for approach_name, check_func in approaches:
            if check_func():
                print(f"✅ {approach_name} available")
                break
        else:
            print(f"❌ No text reading function found")

    except Exception as e:
        print(f"Error reading file: {e}")

except ImportError as e:
    print(f"❌ Failed to import docling: {e}")
    print("Please ensure docling is installed and available")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
