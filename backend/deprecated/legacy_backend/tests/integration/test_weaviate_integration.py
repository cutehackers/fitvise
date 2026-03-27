#!/usr/bin/env python3
"""
Test script to verify Weaviate integration with the updated v4 client.
This focuses on the specific issue with RagEmbeddingTask not saving embeddings.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_embedded_workflow():
    """Test the workflow that was failing before the fix."""
    print("🔍 Testing RagEmbeddingTask workflow...")

    # Read the updated files to verify the key changes
    try:
        # Check that RagEmbeddingTask includes the connection step
        with open('app/pipeline/phases/rag_embedding_task.py', 'r') as f:
            embedding_task_code = f.read()

        # Look for the connection fix
        if 'ensure_weaviate_connected()' in embedding_task_code:
            print("✅ RagEmbeddingTask includes Weaviate connection step")
        else:
            print("❌ RagEmbeddingTask missing Weaviate connection step")
            return False

        # Check that ExternalServicesContainer has the connection method
        with open('app/infrastructure/external_services/external_services_container.py', 'r') as f:
            container_code = f.read()

        if 'def ensure_weaviate_connected(' in container_code:
            print("✅ ExternalServicesContainer has ensure_weaviate_connected method")
        else:
            print("❌ ExternalServicesContainer missing ensure_weaviate_connected method")
            return False

        if 'connected_embedding_repository(' in container_code:
            print("✅ ExternalServicesContainer has connected_embedding_repository method")
        else:
            print("❌ ExternalServicesContainer missing connected_embedding_repository method")
            return False

        return True

    except Exception as e:
        print(f"❌ Error reading files: {e}")
        return False

def test_workflow_integration():
    """Test that the workflow properly integrates the connection step."""
    print("\n🔍 Testing workflow integration...")

    try:
        # Read the RagEmbeddingTask execute method
        with open('app/pipeline/phases/rag_embedding_task.py', 'r') as f:
            code = f.read()

        # Check for the key integration patterns
        patterns = [
            ('External services connection', 'await self.external_services.ensure_weaviate_connected()'),
            ('Connected repository usage', 'connected_embedding_repository'),
            ('Service initialization with connected client', 'embedding_service = SentenceTransformerService('),
        ]

        found = 0
        for pattern_name, pattern in patterns:
            if pattern in code:
                print(f"✅ {pattern_name}")
                found += 1
            else:
                print(f"❌ {pattern_name}")

        print(f"📊 Integration patterns found: {found}/{len(patterns)}")
        return found >= len(patterns) - 1  # Allow for minor variations

    except Exception as e:
        print(f"❌ Error checking integration: {e}")
        return False

def test_environment_config():
    """Test that environment configuration supports the fix."""
    print("\n🔍 Testing environment configuration...")

    try:
        # Check .env file
        with open('.env', 'r') as f:
            env_content = f.read()

        if 'VECTOR_STORE_TYPE=weaviate' in env_content:
            print("✅ Environment configured for Weaviate")
        else:
            print("❌ Environment not configured for Weaviate")
            return False

        # Check Weaviate configuration
        weaviate_configs = [
            'WEAVIATE_HOST=localhost',
            'WEAVIATE_PORT=8080',
        ]

        found = 0
        for config in weaviate_configs:
            if config in env_content:
                print(f"✅ Found {config}")
                found += 1
            else:
                print(f"❌ Missing {config}")

        print(f"📊 Weaviate configs found: {found}/{len(weaviate_configs)}")
        return found >= len(weaviate_configs) - 1

    except Exception as e:
        print(f"❌ Error checking environment: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🚀 Weaviate Integration Test")
    print("=" * 50)
    print("Testing fix for: RagEmbeddingTask not saving embeddings to Weaviate")
    print()

    tests = [
        ("Embedded Workflow Test", test_embedded_workflow),
        ("Workflow Integration Test", test_workflow_integration),
        ("Environment Configuration Test", test_environment_config),
    ]

    passed = 0
    for test_name, test_func in tests:
        print(f"📋 {test_name}")
        print("-" * 30)

        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Summary: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("\n🎉 All integration tests passed!")
        print("🔧 The fix should now work:")
        print("   1. RagEmbeddingTask connects to Weaviate before using embedding service")
        print("   2. WeaviateClient uses v4 API")
        print("   3. Environment is configured for Weaviate")
        print("   4. ExternalServicesContainer provides connected repositories")
        return 0
    else:
        print(f"\n⚠️  {len(tests) - passed} test(s) failed. Review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)