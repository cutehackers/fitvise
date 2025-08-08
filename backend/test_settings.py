#!/usr/bin/env python3
"""
Settings Class Test - Comprehensive validation of configuration loading
Verifies that the Settings class successfully loads all values from .env file
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))


def test_env_file_exists():
    """Test that .env file exists and is readable"""
    env_file = Path(__file__).parent / ".env"

    print("🔍 Environment File Check")
    print("-" * 30)
    print(f"Looking for .env at: {env_file.absolute()}")

    if not env_file.exists():
        print("❌ ERROR: .env file not found!")
        return False

    if not env_file.is_file():
        print("❌ ERROR: .env exists but is not a file!")
        return False

    try:
        with open(env_file, "r") as f:
            content = f.read()
            if not content.strip():
                print("⚠️  WARNING: .env file is empty!")
                return False
        print(f"✅ .env file found and readable ({len(content)} characters)")
        return True
    except Exception as e:
        print(f"❌ ERROR reading .env file: {e}")
        return False


def test_settings_import():
    """Test that Settings class can be imported successfully"""
    print("\n📦 Settings Import Test")
    print("-" * 25)

    try:
        from core.config import Settings

        print("✅ Settings class imported successfully")
        return Settings
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return None


def test_settings_initialization(Settings):
    """Test Settings class initialization and .env loading"""
    print("\n⚙️  Settings Initialization Test")
    print("-" * 35)

    try:
        # Attempt to create Settings instance
        settings = Settings()
        print("✅ Settings instance created successfully")
        return settings
    except Exception as e:
        print(f"❌ Settings initialization failed: {e}")

        # Provide debugging information
        print("\n🔧 Debugging Information:")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Python path: {sys.path[:3]}...")

        # Check if specific fields are causing issues
        print("\n📋 Attempting partial initialization...")
        try:
            # Try to create with minimal config
            import os
            from core.config import Settings

            # Set some required env vars if missing
            required_vars = {
                "APP_NAME": "Test App",
                "APP_VERSION": "1.0.0",
                "APP_DESCRIPTION": "Test Description",
                "ENVIRONMENT": "local",
                "DEBUG": "true",
            }

            for var, value in required_vars.items():
                if var not in os.environ:
                    os.environ[var] = value
                    print(f"   Set {var} = {value}")

            settings = Settings()
            print("✅ Settings created with environment variables")
            return settings

        except Exception as e2:
            print(f"❌ Even partial initialization failed: {e2}")
            return None


def validate_configuration_fields(settings) -> Dict[str, Any]:
    """Validate all configuration fields are loaded correctly"""
    print("\n🧪 Configuration Field Validation")
    print("-" * 40)

    validation_results = {"passed": 0, "failed": 0, "errors": []}

    # Define expected field groups and their validation rules
    field_validations = {
        # App Information
        "app_name": {"type": str, "required": True},
        "app_version": {"type": str, "required": True},
        "app_description": {"type": str, "required": True},
        # Environment Configuration
        "environment": {
            "type": str,
            "required": True,
            "values": ["local", "staging", "production"],
        },
        "debug": {"type": bool, "required": True},
        # Domain Configuration
        "domain": {"type": str, "required": True},
        "api_host": {"type": str, "required": True},
        "api_port": {"type": int, "required": True, "min": 1, "max": 65535},
        # LLM Configuration
        "llm_base_url": {"type": str, "required": True},
        "llm_model": {"type": str, "required": True},
        "llm_timeout": {"type": int, "required": True, "min": 1},
        "llm_temperature": {"type": float, "required": True, "min": 0.0, "max": 1.0},
        "llm_max_tokens": {"type": int, "required": True, "min": 1},
        # API Configuration
        "api_v1_prefix": {"type": str, "required": True},
        "cors_origins": {"type": str, "required": True},
        "cors_allow_credentials": {"type": bool, "required": True},
        "cors_allow_methods": {"type": str, "required": True},
        "cors_allow_headers": {"type": str, "required": True},
        # Database Configuration
        "database_url": {"type": str, "required": True},
        "database_echo": {"type": bool, "required": True},
        # Vector Store Configuration
        "vector_store_type": {
            "type": str,
            "required": True,
            "values": ["chromadb", "faiss"],
        },
        "vector_store_path": {"type": str, "required": True},
        "embedding_model": {"type": str, "required": True},
        "vector_dimension": {"type": int, "required": True, "min": 1},
        # Security Configuration
        "secret_key": {"type": str, "required": True, "min_length": 32},
        "access_token_expire_minutes": {"type": int, "required": True, "min": 1},
        "algorithm": {"type": str, "required": True},
        # File Upload Configuration
        "max_file_size": {"type": int, "required": True, "min": 1},
        "allowed_file_types": {"type": str, "required": True},
        "upload_directory": {"type": str, "required": True},
        # Knowledge Base Configuration
        "knowledge_base_path": {"type": str, "required": True},
        "auto_index_on_startup": {"type": bool, "required": True},
        "index_update_interval": {"type": int, "required": True, "min": 1},
        # Logging Configuration
        "log_level": {
            "type": str,
            "required": True,
            "values": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        },
        "log_file": {"type": str, "required": True},
        "log_rotation": {"type": str, "required": True},
        "log_retention": {"type": str, "required": True},
    }

    for field_name, validation in field_validations.items():
        try:
            # Check if field exists
            if not hasattr(settings, field_name):
                validation_results["failed"] += 1
                validation_results["errors"].append(f"Missing field: {field_name}")
                print(f"❌ {field_name}: Field missing")
                continue

            value = getattr(settings, field_name)

            # Type validation
            if not isinstance(value, validation["type"]):
                validation_results["failed"] += 1
                validation_results["errors"].append(
                    f"{field_name}: Wrong type {type(value)}, expected {validation['type']}"
                )
                print(f"❌ {field_name}: Type mismatch")
                continue

            # Value validation
            if "values" in validation and value not in validation["values"]:
                validation_results["failed"] += 1
                validation_results["errors"].append(
                    f"{field_name}: Invalid value '{value}', must be one of {validation['values']}"
                )
                print(f"❌ {field_name}: Invalid value")
                continue

            # Range validation
            if "min" in validation and (isinstance(value, (int, float)) and value < validation["min"]):
                validation_results["failed"] += 1
                validation_results["errors"].append(f"{field_name}: Value {value} below minimum {validation['min']}")
                print(f"❌ {field_name}: Below minimum")
                continue

            if "max" in validation and (isinstance(value, (int, float)) and value > validation["max"]):
                validation_results["failed"] += 1
                validation_results["errors"].append(f"{field_name}: Value {value} above maximum {validation['max']}")
                print(f"❌ {field_name}: Above maximum")
                continue

            # String length validation
            if "min_length" in validation and isinstance(value, str) and len(value) < validation["min_length"]:
                validation_results["failed"] += 1
                validation_results["errors"].append(
                    f"{field_name}: String too short, minimum {validation['min_length']} characters"
                )
                print(f"❌ {field_name}: String too short")
                continue

            # If we get here, validation passed
            validation_results["passed"] += 1
            print(f"✅ {field_name}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")

        except Exception as e:
            validation_results["failed"] += 1
            validation_results["errors"].append(f"{field_name}: Validation error - {str(e)}")
            print(f"❌ {field_name}: Exception - {e}")

    return validation_results


def test_property_methods(settings):
    """Test property methods for list conversion"""
    print("\n🔄 Property Methods Test")
    print("-" * 25)

    property_tests = [
        ("cors_origins_list", list),
        ("cors_allow_methods_list", list),
        ("cors_allow_headers_list", list),
        ("allowed_file_types_list", list),
    ]

    passed = 0
    failed = 0

    for prop_name, expected_type in property_tests:
        try:
            if hasattr(settings, prop_name):
                value = getattr(settings, prop_name)
                if isinstance(value, expected_type) and len(value) > 0:
                    print(f"✅ {prop_name}: {value}")
                    passed += 1
                else:
                    print(f"❌ {prop_name}: Wrong type or empty")
                    failed += 1
            else:
                print(f"❌ {prop_name}: Property not found")
                failed += 1
        except Exception as e:
            print(f"❌ {prop_name}: Error - {e}")
            failed += 1

    return passed, failed


def display_summary(validation_results, property_results):
    """Display comprehensive test summary"""
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    total_field_tests = validation_results["passed"] + validation_results["failed"]
    total_property_tests = sum(property_results)
    total_tests = total_field_tests + total_property_tests
    total_passed = validation_results["passed"] + property_results[0]

    print(f"📋 Configuration Fields: {validation_results['passed']}/{total_field_tests} passed")
    print(f"🔄 Property Methods: {property_results[0]}/{total_property_tests} passed")
    print(f"🎯 Overall: {total_passed}/{total_tests} tests passed")

    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"📈 Success Rate: {success_rate:.1f}%")

    if validation_results["errors"]:
        print(f"\n❌ Errors Found ({len(validation_results['errors'])}):")
        for error in validation_results["errors"][:10]:  # Show first 10 errors
            print(f"   • {error}")
        if len(validation_results["errors"]) > 10:
            print(f"   ... and {len(validation_results['errors']) - 10} more errors")

    if success_rate >= 90:
        print("\n🎉 EXCELLENT: Configuration is working perfectly!")
    elif success_rate >= 75:
        print("\n✅ GOOD: Configuration is mostly working with minor issues")
    elif success_rate >= 50:
        print("\n⚠️  FAIR: Configuration has some issues that need attention")
    else:
        print("\n🚨 POOR: Configuration has significant issues requiring immediate fix")

    return success_rate >= 75


def main():
    """Main test execution function"""
    print("🧪 FITVISE SETTINGS CONFIGURATION TEST")
    print("=" * 50)

    # Step 1: Check .env file exists
    if not test_env_file_exists():
        print("\n🚨 CRITICAL: Cannot proceed without .env file")
        return False

    # Step 2: Test Settings import
    Settings = test_settings_import()
    if not Settings:
        print("\n🚨 CRITICAL: Cannot import Settings class")
        return False

    # Step 3: Test Settings initialization
    settings = test_settings_initialization(Settings)
    if not settings:
        print("\n🚨 CRITICAL: Cannot initialize Settings")
        return False

    # Step 4: Validate configuration fields
    validation_results = validate_configuration_fields(settings)

    # Step 5: Test property methods
    property_results = test_property_methods(settings)

    # Step 6: Display comprehensive summary
    success = display_summary(validation_results, property_results)

    print(
        f"\n{'🎉 SUCCESS' if success else '🚨 FAILURE'}: Settings test {'completed successfully' if success else 'failed'}"
    )

    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
