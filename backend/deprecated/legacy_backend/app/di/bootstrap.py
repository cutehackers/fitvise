from app.di.containers.container import AppContainer
from app.infrastructure.external_services.vector_stores.weaviate_client import WeaviateClient


def ensure_weaviate_connected(container: AppContainer) -> WeaviateClient:
    client: WeaviateClient = container.weaviate_client()
    
    if not client.is_connected:
        try:
            client.connect()
        except Exception as ex:
            raise RuntimeError(f"Failed to connect Weaviate client: {str(ex)}")

    return client