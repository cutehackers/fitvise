"""
Sample data for testing various scenarios.
"""

from typing import Any, Dict, List

# Sample workout prompts for testing
SAMPLE_WORKOUT_PROMPTS = [
    {
        "prompt": "Create a 30-minute cardio workout for beginners",
        "context": "User is new to fitness, has no equipment, wants to exercise at home",
        "max_tokens": 500,
        "temperature": 0.7,
    },
    {
        "prompt": "Design a strength training routine for intermediate level",
        "context": "User has access to dumbbells and resistance bands, 3 days per week",
        "max_tokens": 750,
        "temperature": 0.5,
    },
    {
        "prompt": "Plan a yoga sequence for flexibility and relaxation",
        "context": "Evening routine, 20 minutes, suitable for all levels",
        "max_tokens": 600,
        "temperature": 0.8,
    },
]

# Sample workout responses
SAMPLE_WORKOUT_RESPONSES = [
    {
        "response": """Here's a 30-minute beginner cardio workout you can do at home:

**Warm-up (5 minutes)**
- Marching in place: 2 minutes
- Arm circles: 1 minute
- Light stretching: 2 minutes

**Main Workout (20 minutes)**
- Jumping jacks: 3 sets of 30 seconds
- High knees: 3 sets of 30 seconds  
- Bodyweight squats: 3 sets of 15 reps
- Push-ups (modified if needed): 3 sets of 10 reps
- Mountain climbers: 3 sets of 20 seconds
- Rest between sets: 30-60 seconds

**Cool-down (5 minutes)**
- Walking in place: 2 minutes
- Static stretches: 3 minutes

Stay hydrated and listen to your body!""",
        "tokens_used": 180,
        "response_time": 1.2,
        "model": "llama2",
        "success": True,
    },
    {
        "response": """Intermediate Strength Training Routine (3 days/week):

**Day 1: Upper Body Focus**
- Dumbbell chest press: 3x10-12
- Bent-over rows: 3x10-12
- Overhead press: 3x8-10
- Bicep curls: 3x12-15
- Tricep extensions: 3x12-15
- Resistance band pull-aparts: 3x15

**Day 2: Lower Body Focus**
- Goblet squats: 3x12-15
- Romanian deadlifts: 3x10-12
- Lunges: 3x10 each leg
- Calf raises: 3x15-20
- Resistance band glute bridges: 3x15
- Wall sit: 3x30-45 seconds

**Day 3: Full Body Circuit**
- Dumbbell thrusters: 3x8
- Renegade rows: 3x6 each arm
- Bulgarian split squats: 3x8 each leg
- Push-ups: 3x8-12
- Plank: 3x30-45 seconds

Rest 48 hours between sessions. Progress by increasing weight or reps weekly.""",
        "tokens_used": 220,
        "response_time": 1.8,
        "model": "llama2",
        "success": True,
    },
]

# Sample health check responses
SAMPLE_HEALTH_RESPONSES = [
    {"status": "healthy", "model": "llama2", "response_time": 0.15},
    {"status": "unhealthy", "model": "llama2", "response_time": 5.0},
]

# Sample error responses
SAMPLE_ERROR_RESPONSES = [
    {"detail": "LLM service is currently unavailable", "status_code": 503},
    {"detail": "Invalid request format", "status_code": 422},
    {"detail": "Request timeout", "status_code": 408},
]

# Sample configuration data
SAMPLE_CONFIG_DATA = {
    "valid_config": {
        "APP_NAME": "Fitvise Test",
        "APP_VERSION": "1.0.0",
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "API_HOST": "localhost",
        "API_PORT": "8000",
        "LLM_BASE_URL": "http://localhost:11434",
        "LLM_MODEL": "llama2",
        "DATABASE_URL": "sqlite:///./test.db",
        "SECRET_KEY": "test-secret-key-minimum-32-characters",
    },
    "invalid_config": {
        "APP_NAME": "",  # Invalid: empty string
        "API_PORT": "99999",  # Invalid: port out of range
        "SECRET_KEY": "short",  # Invalid: too short
    },
}

# Sample API request/response pairs for integration testing
SAMPLE_API_SCENARIOS = [
    {
        "name": "successful_workout_generation",
        "request": {
            "method": "POST",
            "url": "/api/v1/workout/prompt",
            "json": SAMPLE_WORKOUT_PROMPTS[0],
        },
        "expected_response": {
            "status_code": 200,
            "json_contains": [
                "response",
                "tokens_used",
                "response_time",
                "model",
                "success",
            ],
        },
    },
    {
        "name": "health_check",
        "request": {"method": "GET", "url": "/api/v1/workout/health"},
        "expected_response": {
            "status_code": 200,
            "json_contains": ["status", "model", "response_time"],
        },
    },
    {
        "name": "invalid_request_format",
        "request": {
            "method": "POST",
            "url": "/api/v1/workout/prompt",
            "json": {"invalid_field": "invalid_value"},
        },
        "expected_response": {"status_code": 422},
    },
]

# Load test data
LOAD_TEST_DATA = {
    "concurrent_requests": 10,
    "total_requests": 100,
    "request_delay": 0.1,
    "timeout_threshold": 5.0,
    "success_rate_threshold": 0.95,
}

# E2E test scenarios
E2E_SCENARIOS = [
    {
        "name": "complete_workout_generation_flow",
        "description": "User requests a workout and receives a complete response",
        "steps": [
            {"action": "check_service_health", "expected": "healthy"},
            {"action": "submit_workout_request", "data": SAMPLE_WORKOUT_PROMPTS[0]},
            {"action": "verify_response_format", "fields": ["response", "tokens_used"]},
            {"action": "verify_response_content", "min_length": 50},
        ],
    },
    {
        "name": "error_handling_flow",
        "description": "System handles errors gracefully",
        "steps": [
            {"action": "submit_invalid_request", "data": {"invalid": "data"}},
            {"action": "verify_error_response", "status_code": 422},
            {"action": "verify_service_recovery", "expected": "healthy"},
        ],
    },
]
