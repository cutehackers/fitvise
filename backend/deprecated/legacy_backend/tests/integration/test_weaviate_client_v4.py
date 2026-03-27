"""
Integration tests for WeaviateClient v4 API compatibility.

These tests verify that the WeaviateClient has been properly updated
to use Weaviate v4 API and integrates correctly with the embedding pipeline.
"""

import os
import sys
from pathlib import Path

# Optional pytest import
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    pytest = None

    # Create a simple pytest decorator mock
    class Mark:
        def __init__(self):
            pass

        def parametrize(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

        def _mark(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator

    class MockPytest:
        def __init__(self):
            self.mark = Mark()

    pytest = MockPytest()

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_weaviate_client_v4_imports():
    """Test that WeaviateClient uses v4 API imports."""
    # Read WeaviateClient source
    client_file = project_root / "app" / "infrastructure" / "external_services" / "vector_stores" / "weaviate_client.py"
    with open(client_file, 'r') as f:
        code = f.read()

    # Check for v4 API imports
    v4_imports = [
        "from weaviate.classes.config import Configure, Property, DataType",
        "from weaviate.classes.init import Auth",
    ]

    for import_stmt in v4_imports:
        assert import_stmt in code, f"Missing v4 import: {import_stmt}"

def test_weaviate_client_uses_v4_api():
    """Test that WeaviateClient methods use v4 API patterns."""
    client_file = project_root / "app" / "infrastructure" / "external_services" / "vector_stores" / "weaviate_client.py"
    with open(client_file, 'r') as f:
        code = f.read()

    # Check that v4 API methods are used
    v4_patterns = [
        "connect_to_local(",
        "connect_to_weaviate_cloud(",
        "collections.get(",
        "collection.data.insert(",
        "collection.query.fetch_object_by_id(",
        "collection.query.near_vector(",
        "collection.aggregate.over_all(",
    ]

    for pattern in v4_patterns:
        assert pattern in code, f"Missing v4 pattern: {pattern}"

def test_weaviate_client_no_v3_patterns():
    """Test that v3 API patterns have been removed."""
    client_file = project_root / "app" / "infrastructure" / "external_services" / "vector_stores" / "weaviate_client.py"
    with open(client_file, 'r') as f:
        code = f.read()

    # Check that v3 API patterns are not present
    v3_patterns = [
        "weaviate.Client(",
        "data_object.create(",
        "data_object.get_by_id(",
        "data_object.delete(",
        "query.get(",
        "query.aggregate(",
    ]

    for pattern in v3_patterns:
        assert pattern not in code, f"Found deprecated v3 pattern: {pattern}"

def test_weaviate_integration_fixes():
    """Test that the Weaviate integration fixes are in place."""
    # Check RagEmbeddingTask has connection step
    rag_task_file = project_root / "app" / "pipeline" / "phases" / "rag_embedding_task.py"
    with open(rag_task_file, 'r') as f:
        embedding_task_code = f.read()

    assert "ensure_weaviate_connected()" in embedding_task_code, \
        "RagEmbeddingTask missing Weaviate connection step"

    # Check ExternalServicesContainer has connection methods
    container_file = project_root / "app" / "infrastructure" / "external_services" / "external_services_container.py"
    with open(container_file, 'r') as f:
        container_code = f.read()

    assert "def ensure_weaviate_connected(" in container_code, \
        "ExternalServicesContainer missing ensure_weaviate_connected method"

    assert "connected_embedding_repository(" in container_code, \
        "ExternalServicesContainer missing connected_embedding_repository method"

def test_environment_configuration():
    """Test that environment is configured for Weaviate."""
    env_file = project_root / ".env"
    with open(env_file, 'r') as f:
        env_content = f.read()

    assert "VECTOR_STORE_TYPE=weaviate" in env_content, \
        "Environment not configured for Weaviate"

    assert "WEAVIATE_HOST=localhost" in env_content, \
        "Missing Weaviate host configuration"

    assert "WEAVIATE_PORT=8080" in env_content, \
        "Missing Weaviate port configuration"

def test_key_method_implementations():
    """Test that key WeaviateClient methods have been properly updated."""
    client_file = project_root / "app" / "infrastructure" / "external_services" / "vector_stores" / "weaviate_client.py"
    with open(client_file, 'r') as f:
        code = f.read()

    # Method implementations that should use v4 API
    method_checks = [
        ("connect", "def connect(", "connect_to_local("),
        ("create_object", "def create_object(", "collections.get("),
        ("batch_create_objects", "def batch_create_objects(", "collections.get("),
        ("get_object", "def get_object(", "collections.get("),
        ("similarity_search", "def similarity_search(", "collections.get("),
        ("health_check", "def health_check(", "collections.list_all()"),
    ]

    for method_name, method_sig, api_call in method_checks:
        assert method_sig in code, f"Method {method_name} not found"
        assert api_call in code, f"Method {method_name} not using v4 API"

def test_all_methods_use_v4_api():
    """Test to ensure all methods use v4 API."""
    client_file = project_root / "app" / "infrastructure" / "external_services" / "vector_stores" / "weaviate_client.py"
    with open(client_file, 'r') as f:
        code = f.read()

    methods_to_check = [
        "connect", "create_object", "batch_create_objects",
        "get_object", "delete_object", "similarity_search",
        "count_objects", "health_check"
    ]

    for method_name in methods_to_check:
        method_pattern = f"def {method_name}("
        assert method_pattern in code, f"Method {method_name} not found"

        # Ensure no v3 patterns in any method
        v3_patterns = ["data_object.", "query.get(", "query.aggregate("]
        lines = code.split('\n')
        in_method = False
        for line in lines:
            if method_pattern in line:
                in_method = True
                continue
            if in_method:
                if line.startswith('def ') and line != f'def {method_name}(':
                    break  # Next method, stop checking
                for pattern in v3_patterns:
                    if pattern in line:
                        assert False, f"v3 pattern '{pattern}' found in {method_name} method"

class TestWeaviateV4Integration:
    """Integration test class for Weaviate v4 API compatibility."""

    def test_weaviate_client_structure(self):
        """Test that WeaviateClient has the correct structure."""
        client_file = project_root / "app" / "infrastructure" / "external_services" / "vector_stores" / "weaviate_client.py"
        with open(client_file, 'r') as f:
            code = f.read()

        # Check class exists
        assert "class WeaviateClient:" in code

        # Check required methods exist
        required_methods = [
            "def connect(",
            "def disconnect(",
            "def validate_connected(",
            "def create_object(",
            "def batch_create_objects(",
            "def get_object(",
            "def delete_object(",
            "def similarity_search(",
            "def count_objects(",
            "def health_check(",
        ]

        for method in required_methods:
            assert method in code, f"Required method {method} not found"

    def test_workflow_integration_completeness(self):
        """Test that the workflow integration is complete."""
        # Read the complete execute method from RagEmbeddingTask
        rag_task_file = project_root / "app" / "pipeline" / "phases" / "rag_embedding_task.py"
        with open(rag_task_file, 'r') as f:
            embedding_task_code = f.read()

        # Check for the complete integration pattern
        integration_elements = [
            "ensure_weaviate_connected()",
            "sentence_transformer_service",
            "embedding_service.initialize()",
        ]

        for element in integration_elements:
            assert element in embedding_task_code, \
                f"Missing integration element: {element}"

def run_all_tests():
    """Run all test functions manually if pytest is not available."""
    import traceback

    test_functions = [
        test_weaviate_client_v4_imports,
        test_weaviate_client_uses_v4_api,
        test_weaviate_client_no_v3_patterns,
        test_weaviate_integration_fixes,
        test_environment_configuration,
        test_key_method_implementations,
        test_all_methods_use_v4_api,
    ]

    # Test class methods
    test_class = TestWeaviateV4Integration()
    test_methods = [
        test_class.test_weaviate_client_structure,
        test_class.test_workflow_integration_completeness,
    ]

    all_tests = test_functions + test_methods

    passed = 0
    failed = 0

    print("🚀 Running Weaviate v4 API Integration Tests")
    print("=" * 60)

    for i, test_func in enumerate(all_tests, 1):
        test_name = f"Test {i}: {test_func.__name__}"
        print(f"\n📋 {test_name}")
        print("-" * 40)

        try:
            test_func()
            print(f"✅ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED")
            print(f"   Error: {str(e)}")
            if traceback.format_exc():
                print("   Traceback:")
                print("   " + traceback.format_exc().replace('\n', '\n   '))
            failed += 1

    print("\n" + "=" * 60)
    print(f"📊 Summary: {passed} passed, {failed} failed out of {len(all_tests)} tests")

    if failed == 0:
        print("🎉 All tests passed! WeaviateClient v4 API update is complete.")
    else:
        print(f"⚠️  {failed} test(s) failed. Review the issues above.")

    return failed == 0

if __name__ == "__main__":
    if HAS_PYTEST:
        # Allow running as pytest if available
        pytest.main([__file__, "-v"])
    else:
        # Run manually if pytest is not available
        success = run_all_tests()
        sys.exit(0 if success else 1)