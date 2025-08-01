"""
End-to-end tests for complete workout generation workflows.

Tests complete user journeys through the workout generation system.
"""

import pytest
from httpx import AsyncClient
import asyncio

from tests.fixtures.sample_data import E2E_SCENARIOS, SAMPLE_WORKOUT_PROMPTS
from tests.utils.test_helpers import assert_helpers


class TestWorkoutWorkflows:
    """Test complete workout generation workflows."""
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_workout_generation_flow(self, async_test_client: AsyncClient):
        """Test the complete workflow from health check to workout generation."""
        
        # Step 1: Check service health
        health_response = await async_test_client.get("/api/v1/workout/health")
        assert health_response.status_code in [200, 503]  # May be unhealthy in test
        
        health_data = health_response.json()
        service_available = health_data.get("status") == "healthy"
        
        if not service_available:
            pytest.skip("LLM service not available for E2E testing")
        
        # Step 2: Submit workout request
        request_data = SAMPLE_WORKOUT_PROMPTS[0]
        workout_response = await async_test_client.post(
            "/api/v1/workout/prompt", 
            json=request_data
        )
        
        assert workout_response.status_code == 200
        workout_data = workout_response.json()
        
        # Step 3: Verify response format and content
        assert_helpers.assert_valid_workout_response(workout_data)
        assert workout_data["success"] is True
        assert len(workout_data["response"]) >= 50  # Minimum response length
        
        # Step 4: Verify response contains workout-related content
        response_text = workout_data["response"].lower()
        workout_keywords = ["workout", "exercise", "minutes", "sets", "reps", "cardio", "strength"]
        assert any(keyword in response_text for keyword in workout_keywords), \
            "Response should contain workout-related content"
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_multiple_workout_requests_flow(self, async_test_client: AsyncClient):
        """Test generating multiple different workouts in sequence."""
        
        # Check service availability
        health_response = await async_test_client.get("/api/v1/workout/health")
        if health_response.status_code != 200:
            pytest.skip("LLM service not available for E2E testing")
        
        # Test different workout types
        workout_requests = [
            {"prompt": "30-minute beginner cardio workout", "max_tokens": 400},
            {"prompt": "Upper body strength training routine", "max_tokens": 500},
            {"prompt": "Yoga flow for flexibility", "max_tokens": 350}
        ]
        
        responses = []
        
        for request_data in workout_requests:
            response = await async_test_client.post(
                "/api/v1/workout/prompt",
                json=request_data
            )
            
            assert response.status_code == 200
            response_data = response.json()
            assert_helpers.assert_valid_workout_response(response_data)
            responses.append(response_data)
        
        # Verify all responses are different and relevant
        response_texts = [r["response"] for r in responses]
        
        # Responses should be different
        assert len(set(response_texts)) == len(response_texts), "All responses should be unique"
        
        # Each response should be relevant to its request
        assert "cardio" in response_texts[0].lower() or "aerobic" in response_texts[0].lower()
        assert "strength" in response_texts[1].lower() or "muscle" in response_texts[1].lower()
        assert "yoga" in response_texts[2].lower() or "flexibility" in response_texts[2].lower()
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_error_recovery_flow(self, async_test_client: AsyncClient):
        """Test system recovery from errors."""
        
        # Step 1: Make an invalid request
        invalid_response = await async_test_client.post(
            "/api/v1/workout/prompt",
            json={"invalid_field": "invalid_value"}
        )
        
        assert invalid_response.status_code == 422
        error_data = invalid_response.json()
        assert "detail" in error_data
        
        # Step 2: Verify service is still healthy after error
        health_response = await async_test_client.get("/api/v1/workout/health")
        assert health_response.status_code in [200, 503]
        
        # Step 3: Make a valid request to ensure recovery
        if health_response.status_code == 200:
            valid_request = {"prompt": "Simple 15-minute workout"}
            valid_response = await async_test_client.post(
                "/api/v1/workout/prompt",
                json=valid_request
            )
            
            # Should work normally after error
            if valid_response.status_code == 200:  # Service may be unavailable
                response_data = valid_response.json()
                assert_helpers.assert_valid_workout_response(response_data)
    
    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_concurrent_users_flow(self, async_test_client: AsyncClient):
        """Test multiple concurrent users requesting workouts."""
        
        # Check service availability
        health_response = await async_test_client.get("/api/v1/workout/health")
        if health_response.status_code != 200:
            pytest.skip("LLM service not available for E2E testing")
        
        # Simulate concurrent users
        async def user_workout_request(user_id: int):
            request_data = {
                "prompt": f"Personalized workout for user {user_id}",
                "context": f"User {user_id} preferences and fitness level",
                "max_tokens": 300
            }
            
            response = await async_test_client.post(
                "/api/v1/workout/prompt",
                json=request_data
            )
            
            return response.status_code, response.json()
        
        # Execute concurrent requests
        num_concurrent_users = 3
        tasks = [user_workout_request(i) for i in range(num_concurrent_users)]
        results = await asyncio.gather(*tasks)
        
        # Verify all requests succeeded
        for status_code, response_data in results:
            assert status_code == 200, f"Request failed with status {status_code}"
            assert_helpers.assert_valid_workout_response(response_data)
            assert response_data["success"] is True
        
        # Verify responses are personalized (different)
        response_texts = [data["response"] for _, data in results]
        unique_responses = set(response_texts)
        assert len(unique_responses) >= 2, "Responses should show some variation"
    
    @pytest.mark.e2e
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_performance_under_load(self, async_test_client: AsyncClient):
        """Test system performance under sustained load."""
        
        # Check service availability
        health_response = await async_test_client.get("/api/v1/workout/health")
        if health_response.status_code != 200:
            pytest.skip("LLM service not available for performance testing")
        
        import time
        
        # Performance test parameters
        num_requests = 10
        max_response_time = 30.0  # seconds
        min_success_rate = 0.8
        
        async def timed_request():
            start_time = time.time()
            
            request_data = {"prompt": "Quick workout routine", "max_tokens": 200}
            response = await async_test_client.post(
                "/api/v1/workout/prompt",
                json=request_data
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return {
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200
            }
        
        # Execute load test
        start_time = time.time()
        tasks = [timed_request() for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = sum(1 for r in results if r["success"])
        success_rate = successful_requests / num_requests
        avg_response_time = sum(r["response_time"] for r in results) / num_requests
        max_response_time_actual = max(r["response_time"] for r in results)
        
        # Performance assertions
        assert success_rate >= min_success_rate, \
            f"Success rate {success_rate:.2f} below threshold {min_success_rate}"
        
        assert max_response_time_actual <= max_response_time, \
            f"Max response time {max_response_time_actual:.2f}s exceeded limit {max_response_time}s"
        
        assert total_time <= num_requests * 5.0, \
            f"Total test time {total_time:.2f}s too long for {num_requests} requests"
        
        # Log performance metrics for monitoring
        print(f"\nPerformance Test Results:")
        print(f"  Requests: {num_requests}")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Average Response Time: {avg_response_time:.2f}s")
        print(f"  Max Response Time: {max_response_time_actual:.2f}s")
        print(f"  Total Test Time: {total_time:.2f}s")
    
    @pytest.mark.e2e
    @pytest.mark.external
    @pytest.mark.asyncio
    async def test_real_llm_integration(self, async_test_client: AsyncClient):
        """Test integration with actual LLM service (requires external service)."""
        
        # This test requires actual LLM service running
        health_response = await async_test_client.get("/api/v1/workout/health")
        
        if health_response.status_code != 200:
            pytest.skip("External LLM service not available")
        
        health_data = health_response.json()
        if health_data.get("status") != "healthy":
            pytest.skip("External LLM service reports unhealthy status")
        
        # Test with realistic workout request
        request_data = {
            "prompt": "Create a comprehensive 45-minute full-body strength training workout for an intermediate fitness enthusiast",
            "context": "User has access to dumbbells, barbells, and basic gym equipment. Prefers compound movements.",
            "max_tokens": 800,
            "temperature": 0.7
        }
        
        response = await async_test_client.post(
            "/api/v1/workout/prompt",
            json=request_data
        )
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Comprehensive validation for real LLM response
        assert_helpers.assert_valid_workout_response(response_data)
        assert response_data["success"] is True
        
        workout_text = response_data["response"]
        
        # Should be substantial response
        assert len(workout_text) >= 200, "Real LLM should provide detailed response"
        
        # Should contain relevant fitness terminology
        fitness_terms = [
            "workout", "exercise", "sets", "reps", "minutes", 
            "strength", "muscle", "training", "warm", "cool"
        ]
        found_terms = [term for term in fitness_terms if term in workout_text.lower()]
        assert len(found_terms) >= 5, f"Response should contain fitness terminology. Found: {found_terms}"
        
        # Should have reasonable token usage
        assert response_data["tokens_used"] > 0, "Should report token usage"
        assert response_data["tokens_used"] <= request_data["max_tokens"], "Should respect token limit"
        
        # Should have reasonable response time
        assert response_data["response_time"] > 0, "Should report response time"
        assert response_data["response_time"] <= 30.0, "Response time should be reasonable"