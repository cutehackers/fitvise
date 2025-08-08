"""
Example usage of the Workout API endpoints.
"""

import asyncio
import httpx
import json

# API base URL (adjust as needed)
BASE_URL = "http://localhost:8000/api/v1"


async def test_workout_api():
    """Test the workout API endpoints"""

    async with httpx.AsyncClient() as client:

        # Test health endpoint
        print("=== Testing Health Endpoint ===")
        try:
            response = await client.get(f"{BASE_URL}/workout/health")
            print(f"Status: {response.status_code}")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Health check failed: {e}")

        print("\n" + "=" * 50 + "\n")

        # Test fitness AI prompt endpoint
        print("=== Testing Fitness AI Prompt ===")
        workout_request = {
            "query": "Create a 30-minute full body workout for beginners",
            "context": "I'm new to fitness, have basic equipment at home (dumbbells, resistance bands), and want to build strength",
            "temperature": 0.7,
            "max_tokens": 500,
        }

        try:
            response = await client.post(f"{BASE_URL}/workout/prompt", json=workout_request, timeout=30.0)
            print(f"Status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"AI Response:\n{result['response']}")
                print(f"\nModel: {result['model']}")
                print(f"Tokens Used: {result['tokens_used']}")
                print(f"Duration: {result['duration_ms']:.2f}ms" if result["duration_ms"] else "N/A")
            else:
                print(f"Error Response: {json.dumps(response.json(), indent=2)}")

        except Exception as e:
            print(f"Fitness AI prompt failed: {e}")

        print("\n" + "=" * 50 + "\n")

        # Test models endpoint
        print("=== Testing Models Endpoint ===")
        try:
            response = await client.get(f"{BASE_URL}/workout/models")
            print(f"Status: {response.status_code}")
            print(f"Models Info: {json.dumps(response.json(), indent=2)}")
        except Exception as e:
            print(f"Models endpoint failed: {e}")


def test_api_with_curl():
    """Show curl examples for testing the API"""

    print("=== CURL Examples ===\n")

    print("1. Health Check:")
    print(f"curl -X GET '{BASE_URL}/workout/health'")
    print()

    print("2. Send Fitness Prompt:")
    curl_data = {
        "query": "Create a 20-minute HIIT workout",
        "context": "I want to burn calories and improve cardio",
        "temperature": 0.7,
    }

    print(f"curl -X POST '{BASE_URL}/workout/prompt' \\")
    print("  -H 'Content-Type: application/json' \\")
    print(f"  -d '{json.dumps(curl_data)}'")
    print()

    print("3. Get Models Info:")
    print(f"curl -X GET '{BASE_URL}/workout/models'")
    print()


if __name__ == "__main__":
    print("=== Workout API Testing ===\n")

    # Show curl examples
    test_api_with_curl()

    print("=== Running Async Tests ===")
    # Run async tests
    asyncio.run(test_workout_api())
