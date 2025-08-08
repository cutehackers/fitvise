"""
Example usage of the LlmService class

This demonstrates how to use the LlmService to handle user queries
and get AI-generated responses from the LLM API.
"""

import asyncio
from app.services.llm_service import LlmService, QueryRequest


async def main():
    """Example usage of LlmService"""

    # Initialize the service
    service = LlmService()

    # Example 1: Basic query
    request = QueryRequest(query="What is machine learning?")
    response = await service.query(request)

    print("=== Basic Query Example ===")
    print(f"Query: {request.query}")
    print(f"Success: {response.success}")
    if response.success:
        print(f"Response: {response.response}")
        print(f"Model: {response.model}")
        print(f"Total tokens: {response.tokens_used}")
        print(f"Prompt tokens: {response.prompt_tokens}")
        print(f"Completion tokens: {response.completion_tokens}")
        print(f"Duration: {response.total_duration_ms:.2f}ms" if response.total_duration_ms else "Duration: N/A")
    else:
        print(f"Error: {response.error}")

    print("\n" + "=" * 50 + "\n")

    # Example 2: Query with context
    request_with_context = QueryRequest(
        query="How can this help with fitness tracking?",
        context="The user is building a fitness application called Fitvise that tracks workouts and provides personalized recommendations.",
        temperature=0.7,
        max_tokens=500,
    )

    response_with_context = await service.query(request_with_context)

    print("=== Query with Context Example ===")
    print(f"Query: {request_with_context.query}")
    print(f"Context: {request_with_context.context}")
    print(f"Success: {response_with_context.success}")
    if response_with_context.success:
        print(f"Response: {response_with_context.response}")
        print(f"Total tokens: {response_with_context.tokens_used}")
        print(
            f"Duration: {response_with_context.total_duration_ms:.2f}ms"
            if response_with_context.total_duration_ms
            else "Duration: N/A"
        )
    else:
        print(f"Error: {response_with_context.error}")

    print("\n" + "=" * 50 + "\n")

    # Example 3: Health check
    health_status = await service.health()
    print(f"LLM Service Health: {'✅ Healthy' if health_status else '❌ Unavailable'}")

    # Clean up
    await service.close()


if __name__ == "__main__":
    asyncio.run(main())
