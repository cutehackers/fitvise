"""ML-based source categorization use case implementation (Task 1.1.3)."""
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.domain.entities.data_source import DataSource
from app.domain.entities.document import Document
from app.domain.repositories.data_source_repository import DataSourceRepository
from app.infrastructure.external_services.ml_services.categorization.sklearn_categorizer import (
    SklearnDocumentCategorizer
)


class CategorizeSourcesRequest:
    """ML categorization request - train model, use synthetic data, set confidence thresholds."""
    
    def __init__(
        self,
        train_model: bool = False,
        training_documents: Optional[List[Dict[str, Any]]] = None,
        use_synthetic_data: bool = True,
        synthetic_data_size: int = 100,
        categorize_sources: bool = True,
        source_ids: Optional[List[str]] = None,
        min_confidence: float = 0.6,
        model_type: str = "logistic_regression",
        save_model: bool = True,
        model_path: Optional[str] = None
    ):
        """Initialize categorization request."""
        self.train_model = train_model
        self.training_documents = training_documents or []
        self.use_synthetic_data = use_synthetic_data
        self.synthetic_data_size = synthetic_data_size
        self.categorize_sources = categorize_sources
        self.source_ids = source_ids
        self.min_confidence = min_confidence
        self.model_type = model_type
        self.save_model = save_model
        self.model_path = model_path


class CategorizeSourcesResponse:
    """Response from ML-based source categorization."""
    
    def __init__(
        self,
        success: bool,
        training_results: Optional[Dict[str, Any]] = None,
        categorization_results: Optional[List[Dict[str, Any]]] = None,
        model_info: Optional[Dict[str, Any]] = None,
        updated_sources: Optional[List[DataSource]] = None,
        error_message: Optional[str] = None
    ):
        """Initialize categorization response."""
        self.success = success
        self.training_results = training_results
        self.categorization_results = categorization_results or []
        self.model_info = model_info
        self.updated_sources = updated_sources or []
        self.error_message = error_message
    
    @property
    def meets_accuracy_target(self) -> bool:
        """Check if model meets 85% accuracy target."""
        if not self.training_results:
            return False
        return self.training_results.get("meets_target_accuracy", False)
    
    @property
    def total_categorized(self) -> int:
        """Get total number of categorized sources."""
        return len(self.categorization_results)


class CategorizeSourcesUseCase:
    """Task 1.1.3 - ML categorization with scikit-learn achieving 85% accuracy on 100+ docs.
    
    Trains models, generates synthetic data, categorizes sources with confidence scoring.
    
    Examples:
        >>> use_case = CategorizeSourcesUseCase(repository, categorizer)
        >>> request = CategorizeSourcesRequest(train_model=True, use_synthetic_data=True)
        >>> response = await use_case.execute(request)
        >>> response.success and response.meets_accuracy_target
        True
        >>> response.training_results["test_accuracy"] >= 0.85
        True
    """
    
    def __init__(
        self,
        data_source_repository: DataSourceRepository,
        categorizer: Optional[SklearnDocumentCategorizer] = None
    ):
        """Initialize the use case."""
        self.repository = data_source_repository
        self.categorizer = categorizer or SklearnDocumentCategorizer()
    
    async def execute(self, request: CategorizeSourcesRequest) -> CategorizeSourcesResponse:
        """Execute the ML-based categorization process."""
        try:
            training_results = None
            categorization_results = []
            updated_sources = []
            
            # 1. Train model if requested
            if request.train_model:
                training_data = request.training_documents.copy()
                
                # Add synthetic training data if requested
                if request.use_synthetic_data:
                    synthetic_data = await self.categorizer.generate_synthetic_training_data(
                        request.synthetic_data_size
                    )
                    training_data.extend(synthetic_data)
                
                if len(training_data) < 10:
                    raise ValueError("Need at least 10 training documents (including synthetic)")
                
                # Train the model
                training_results = await self.categorizer.train_model(
                    training_data,
                    test_size=0.2,
                    cross_validation_folds=5
                )
                
                # Save model if requested
                if request.save_model:
                    model_path = request.model_path or "models/source_categorizer.pkl"
                    await self.categorizer.save_model(model_path)
            
            # 2. Load existing model if not training
            elif not self.categorizer.is_trained:
                model_path = request.model_path or "models/source_categorizer.pkl"
                try:
                    await self.categorizer.load_model(model_path)
                except FileNotFoundError:
                    # Train with synthetic data if no model exists
                    synthetic_data = await self.categorizer.generate_synthetic_training_data(100)
                    training_results = await self.categorizer.train_model(synthetic_data)
            
            # 3. Categorize sources if requested
            if request.categorize_sources:
                # Get sources to categorize
                if request.source_ids:
                    sources = []
                    for source_id in request.source_ids:
                        source = await self.repository.find_by_id(source_id)
                        if source:
                            sources.append(source)
                else:
                    sources = await self.repository.find_all()
                
                # Convert sources to documents for categorization
                source_documents = []
                for source in sources:
                    doc_data = self._convert_source_to_document(source)
                    source_documents.append(doc_data)
                
                if source_documents:
                    # Batch categorize
                    batch_results = await self.categorizer.batch_categorize(
                        source_documents,
                        batch_size=50
                    )
                    
                    # Update sources with categories
                    for i, (source, result) in enumerate(zip(sources, batch_results)):
                        if result["categories"] and result["confidence"] >= request.min_confidence:
                            # Extract category names
                            predicted_categories = [
                                cat["category"] for cat in result["categories"]
                            ]
                            max_confidence = max(cat["confidence"] for cat in result["categories"])
                            
                            # Update source with predicted categories
                            # Note: In a full implementation, you'd add category fields to DataSource
                            # For now, we'll store in the result
                            categorization_results.append({
                                "source_id": str(source.id),
                                "source_name": source.name,
                                "predicted_categories": predicted_categories,
                                "confidence": max_confidence,
                                "category_details": result["categories"]
                            })
                            
                            updated_sources.append(source)
                        else:
                            categorization_results.append({
                                "source_id": str(source.id),
                                "source_name": source.name,
                                "predicted_categories": [],
                                "confidence": result["confidence"],
                                "error": result.get("error") or "Low confidence prediction"
                            })
            
            # Get model info
            model_info = self.categorizer.get_model_info()
            
            return CategorizeSourcesResponse(
                success=True,
                training_results=training_results,
                categorization_results=categorization_results,
                model_info=model_info,
                updated_sources=updated_sources
            )
            
        except Exception as e:
            return CategorizeSourcesResponse(
                success=False,
                error_message=str(e)
            )
    
    def _convert_source_to_document(self, source: DataSource) -> Dict[str, Any]:
        """Convert DataSource to document format for categorization."""
        # Extract text features from data source
        content_parts = []
        
        # Use name and description as primary content
        content_parts.append(source.name)
        content_parts.append(source.description)
        
        # Add source type information
        content_parts.append(f"source type {source.source_type.value}")
        
        # Add connection information if available
        if source.source_info.connection_config.base_url:
            content_parts.append(f"url {source.source_info.connection_config.base_url}")
        
        if source.source_info.connection_config.database_name:
            content_parts.append(f"database {source.source_info.connection_config.database_name}")
        
        # Add tags if available
        if hasattr(source.source_info, 'tags') and source.source_info.tags:
            content_parts.extend(source.source_info.tags)
        
        return {
            "id": str(source.id),
            "title": source.name,
            "content": " ".join(content_parts),
            "description": source.description,
            "filename": f"{source.name.replace(' ', '_')}.{source.source_type.value}",
            "source_type": source.source_type.value
        }
    
    async def evaluate_categorization_accuracy(
        self, 
        test_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate the categorization model accuracy."""
        if not self.categorizer.is_trained:
            raise ValueError("Model is not trained. Cannot evaluate accuracy.")
        
        evaluation_results = await self.categorizer.evaluate_model(test_documents)
        
        return {
            "evaluation_results": evaluation_results,
            "meets_task_requirements": evaluation_results.get("meets_target", False),
            "recommendation": (
                "Model meets 85% accuracy target" if evaluation_results.get("meets_target")
                else "Model needs improvement or more training data"
            )
        }
    
    async def get_categorization_statistics(self) -> Dict[str, Any]:
        """Get statistics about the categorization system."""
        model_info = self.categorizer.get_model_info()
        
        if not self.categorizer.is_trained:
            return {
                "status": "not_ready",
                "message": "Categorization model is not trained",
                "model_info": model_info
            }
        
        # Get all sources for statistics
        all_sources = await self.repository.find_all()
        
        # In a full implementation, you'd track which sources have been categorized
        # For now, provide general statistics
        
        return {
            "status": "ready",
            "model_info": model_info,
            "total_sources": len(all_sources),
            "categorization_ready": True,
            "supported_categories": model_info.get("categories", []),
            "training_accuracy": model_info.get("training_stats", {}).get("test_accuracy"),
            "meets_requirements": (
                model_info.get("training_stats", {}).get("meets_target_accuracy", False)
            )
        }
    
    async def retrain_model_with_feedback(
        self, 
        feedback_documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Retrain the model with user feedback."""
        if len(feedback_documents) < 10:
            raise ValueError("Need at least 10 feedback documents for retraining")
        
        # Combine with synthetic data to ensure sufficient training size
        synthetic_data = await self.categorizer.generate_synthetic_training_data(50)
        training_data = feedback_documents + synthetic_data
        
        # Retrain model
        training_results = await self.categorizer.train_model(
            training_data,
            test_size=0.2,
            cross_validation_folds=5
        )
        
        return {
            "retrain_results": training_results,
            "feedback_documents_used": len(feedback_documents),
            "synthetic_documents_used": len(synthetic_data),
            "new_accuracy": training_results.get("test_accuracy"),
            "improvement": training_results.get("meets_target_accuracy", False)
        }
    
    async def predict_source_categories(
        self, 
        source_id: str, 
        min_confidence: float = 0.6
    ) -> Dict[str, Any]:
        """Predict categories for a single data source."""
        if not self.categorizer.is_trained:
            raise ValueError("Model is not trained")
        
        source = await self.repository.find_by_id(source_id)
        if not source:
            raise ValueError(f"Source not found: {source_id}")
        
        # Convert to document format
        doc_data = self._convert_source_to_document(source)
        
        # Predict categories
        predictions = await self.categorizer.predict_categories(
            doc_data["content"],
            top_k=5,
            min_confidence=min_confidence
        )
        
        return {
            "source_id": source_id,
            "source_name": source.name,
            "predictions": predictions,
            "max_confidence": max([p["confidence"] for p in predictions]) if predictions else 0.0,
            "predicted_at": datetime.utcnow().isoformat()
        }