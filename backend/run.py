# run.py - Main entry point for Fitvise Backend API
import uvicorn

from backend.app.core.settings import settings

if __name__ == "__main__":
    print("🌪️  Starting Fitvise backend server...")
    print(f"🌍 Environment: {settings.environment}")
    print(f"🌐 API will be available at: http://{settings.api_host}:{settings.api_port}")
    print(f"📜 Interactive docs at: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"🔄 Auto-reload: {'enabled' if settings.debug else 'disabled'}")
    print(f"🤖 LLM Model: {settings.llm_model}")
    print("-" * 60)

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,  # Auto-reload during development
        log_level=settings.log_level.lower(),
        access_log=True,
    )
