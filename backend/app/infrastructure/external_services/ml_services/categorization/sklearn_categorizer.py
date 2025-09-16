"""Scikit-learn based document categorization service."""
import pickle
import joblib
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.multioutput import MultiOutputClassifier


class SklearnDocumentCategorizer:
    """ML categorizer - TF-IDF + LogisticRegression/NaiveBayes/RandomForest targeting 85% accuracy.
    
    Multi-label classification with confidence scoring, synthetic data generation & model persistence.
    
    Examples:
        >>> categorizer = SklearnDocumentCategorizer(model_type="logistic_regression")
        >>> training_data = await categorizer.generate_synthetic_training_data(100)
        >>> results = await categorizer.train_model(training_data)
        >>> results["test_accuracy"] >= 0.85
        True
        >>> predictions = await categorizer.predict_categories("financial report quarterly")
        >>> predictions[0]["category"]
        'financial'
    """
    
    def __init__(
        self, 
        model_type: str = "logistic_regression",
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.8,
        model_save_path: Optional[str] = None
    ):
        """Initialize the categorizer."""
        self.model_type = model_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.model_save_path = model_save_path or "models/document_categorizer.pkl"
        
        # Model pipeline
        self.pipeline = None
        self.categories = []
        self.is_trained = False
        self.training_stats = {}
        
        # Predefined categories for business documents
        self.default_categories = [
            "financial", "legal", "hr", "technical", "marketing",
            "research", "policy", "manual", "report", "correspondence"
        ]
        
        # Feature keywords for each category
        self.category_keywords = {
            "financial": [
                "budget", "revenue", "profit", "loss", "expense", "invoice", 
                "financial", "accounting", "balance", "statement", "audit"
            ],
            "legal": [
                "contract", "agreement", "legal", "law", "compliance", "regulation",
                "terms", "conditions", "liability", "rights", "clause"
            ],
            "hr": [
                "employee", "hr", "human resources", "personnel", "hiring", "training",
                "performance", "benefits", "salary", "policy", "onboarding"
            ],
            "technical": [
                "technical", "specification", "documentation", "api", "code", "system",
                "architecture", "design", "implementation", "development", "software"
            ],
            "marketing": [
                "marketing", "campaign", "promotion", "brand", "customer", "market",
                "advertising", "social media", "content", "strategy", "audience"
            ],
            "research": [
                "research", "study", "analysis", "data", "findings", "methodology",
                "hypothesis", "conclusion", "experiment", "survey", "statistics"
            ],
            "policy": [
                "policy", "procedure", "guideline", "standard", "rule", "protocol",
                "governance", "framework", "compliance", "requirement", "process"
            ],
            "manual": [
                "manual", "guide", "instructions", "handbook", "tutorial", "how-to",
                "step-by-step", "user guide", "reference", "documentation", "help"
            ],
            "report": [
                "report", "summary", "overview", "status", "progress", "quarterly",
                "annual", "monthly", "executive", "dashboard", "metrics"
            ],
            "correspondence": [
                "email", "letter", "memo", "message", "communication", "notice",
                "announcement", "update", "notification", "correspondence", "reply"
            ]
        }
    
    def _create_pipeline(self) -> Pipeline:
        """Create the ML pipeline based on model type."""
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        if self.model_type == "naive_bayes":
            classifier = MultinomialNB(alpha=0.1)
        elif self.model_type == "random_forest":
            classifier = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                n_jobs=-1
            )
        else:  # Default to logistic regression
            from sklearn.multiclass import OneVsRestClassifier
            classifier = OneVsRestClassifier(
                LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    n_jobs=-1
                )
            )
        
        return Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
    
    async def train_model(
        self, 
        documents: List[Dict[str, Any]], 
        test_size: float = 0.2,
        cross_validation_folds: int = 5
    ) -> Dict[str, Any]:
        """Train the categorization model."""
        
        if len(documents) < 10:
            raise ValueError("Need at least 10 documents for training")
        
        # Prepare training data
        texts = []
        labels = []
        
        for doc in documents:
            text = self._extract_text_features(doc)
            categories = doc.get('categories', [])
            
            if not text or not categories:
                continue
            
            texts.append(text)
            labels.append(categories)
        
        if len(texts) < 10:
            raise ValueError("Not enough valid training documents")
        
        # Convert to binary multi-label format
        self.categories = list(set([cat for cats in labels for cat in cats]))
        y_binary = self._labels_to_binary(labels, self.categories)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, y_binary, test_size=test_size, random_state=42
        )
        
        # Create and train pipeline
        self.pipeline = self._create_pipeline()
        
        # For multi-label classification, wrap classifier
        multi_classifier = MultiOutputClassifier(self.pipeline)
        
        # Train model
        start_time = datetime.utcnow()
        multi_classifier.fit(X_train, y_train)
        training_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Evaluate model
        y_pred = multi_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        if cross_validation_folds > 1:
            cv_scores = cross_val_score(
                multi_classifier, texts, y_binary, 
                cv=cross_validation_folds, 
                scoring='accuracy',
                n_jobs=-1
            )
        else:
            cv_scores = [accuracy]
        
        # Store the trained model
        self.pipeline = multi_classifier
        self.is_trained = True
        
        # Training statistics
        self.training_stats = {
            "training_time_seconds": training_time,
            "num_documents": len(texts),
            "num_categories": len(self.categories),
            "test_accuracy": accuracy,
            "cv_mean_accuracy": np.mean(cv_scores),
            "cv_std_accuracy": np.std(cv_scores),
            "meets_target_accuracy": accuracy >= 0.85,
            "trained_at": datetime.utcnow().isoformat(),
            "categories": self.categories
        }
        
        # Save model if path provided
        if self.model_save_path:
            await self.save_model(self.model_save_path)
        
        return self.training_stats
    
    async def predict_categories(
        self, 
        text: str, 
        top_k: int = 3,
        min_confidence: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Predict categories for a document."""
        
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train_model() first.")
        
        if not text.strip():
            return []
        
        # Get predictions and probabilities
        text_features = self._extract_text_features({"content": text})
        probabilities = self.pipeline.predict_proba([text_features])[0]
        predictions = self.pipeline.predict([text_features])[0]
        
        # Combine predictions with confidence scores
        results = []
        for i, (category, prob, pred) in enumerate(zip(self.categories, probabilities, predictions)):
            if pred == 1 and prob >= min_confidence:
                results.append({
                    "category": category,
                    "confidence": float(prob),
                    "predicted": bool(pred)
                })
        
        # Sort by confidence and return top-k
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]
    
    async def batch_categorize(
        self, 
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """Categorize multiple documents in batches."""
        
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train_model() first.")
        
        results = []
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_texts = [self._extract_text_features(doc) for doc in batch]
            
            # Filter out empty texts
            valid_indices = [j for j, text in enumerate(batch_texts) if text.strip()]
            valid_texts = [batch_texts[j] for j in valid_indices]
            
            if not valid_texts:
                # Add empty results for this batch
                for doc in batch:
                    results.append({
                        "document_id": doc.get("id"),
                        "categories": [],
                        "confidence": 0.0,
                        "error": "No valid text content"
                    })
                continue
            
            # Predict for valid documents
            try:
                predictions = self.pipeline.predict(valid_texts)
                probabilities = self.pipeline.predict_proba(valid_texts)
                
                batch_results = []
                valid_idx = 0
                
                for j, doc in enumerate(batch):
                    if j in valid_indices:
                        # Get predictions for this document
                        doc_pred = predictions[valid_idx]
                        doc_probs = probabilities[valid_idx]
                        
                        categories = []
                        max_confidence = 0.0
                        
                        for cat_idx, (category, pred, prob) in enumerate(zip(
                            self.categories, doc_pred, doc_probs
                        )):
                            if pred == 1:
                                categories.append({
                                    "category": category,
                                    "confidence": float(prob)
                                })
                                max_confidence = max(max_confidence, prob)
                        
                        batch_results.append({
                            "document_id": doc.get("id"),
                            "categories": categories,
                            "confidence": float(max_confidence),
                            "error": None
                        })
                        
                        valid_idx += 1
                    else:
                        batch_results.append({
                            "document_id": doc.get("id"),
                            "categories": [],
                            "confidence": 0.0,
                            "error": "No valid text content"
                        })
                
                results.extend(batch_results)
                
            except Exception as e:
                # Add error results for this batch
                for doc in batch:
                    results.append({
                        "document_id": doc.get("id"),
                        "categories": [],
                        "confidence": 0.0,
                        "error": str(e)
                    })
        
        return results
    
    def _extract_text_features(self, document: Dict[str, Any]) -> str:
        """Extract text features from document for classification."""
        text_parts = []
        
        # Extract various text fields
        content = document.get("content", "")
        title = document.get("title", "")
        description = document.get("description", "")
        filename = document.get("filename", "")
        keywords = document.get("keywords", [])
        
        # Combine text fields with appropriate weighting
        if title:
            text_parts.append(title * 3)  # Title is important
        
        if description:
            text_parts.append(description * 2)  # Description is moderately important
        
        if filename:
            # Clean filename and give it some weight
            clean_filename = filename.replace("_", " ").replace("-", " ").replace(".", " ")
            text_parts.append(clean_filename * 2)
        
        if isinstance(keywords, list):
            text_parts.append(" ".join(keywords) * 2)
        
        if content:
            text_parts.append(content)
        
        # If no content, try to infer from category keywords
        combined_text = " ".join(text_parts)
        if not combined_text.strip() and filename:
            # Use filename-based categorization
            combined_text = self._enhance_text_with_keywords(filename, combined_text)
        
        return combined_text.strip()
    
    def _enhance_text_with_keywords(self, filename: str, text: str) -> str:
        """Enhance text with category-specific keywords based on filename patterns."""
        filename_lower = filename.lower()
        
        # Check filename against category keywords
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in filename_lower:
                    text += f" {keyword} {category}"
        
        return text
    
    def _labels_to_binary(self, labels: List[List[str]], all_categories: List[str]) -> np.ndarray:
        """Convert string labels to binary multi-label format."""
        binary_labels = np.zeros((len(labels), len(all_categories)))
        
        for i, doc_labels in enumerate(labels):
            for label in doc_labels:
                if label in all_categories:
                    j = all_categories.index(label)
                    binary_labels[i, j] = 1
        
        return binary_labels
    
    async def generate_synthetic_training_data(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Generate synthetic training data for testing."""
        
        synthetic_docs = []
        
        for i in range(num_samples):
            # Randomly select 1-3 categories
            num_cats = np.random.randint(1, 4)
            doc_categories = np.random.choice(
                list(self.category_keywords.keys()), 
                size=num_cats, 
                replace=False
            ).tolist()
            
            # Generate text based on selected categories
            text_parts = []
            for category in doc_categories:
                keywords = np.random.choice(
                    self.category_keywords[category],
                    size=np.random.randint(2, 6),
                    replace=True
                )
                text_parts.extend(keywords)
            
            # Add some random business text
            common_words = [
                "company", "business", "organization", "team", "project", 
                "management", "process", "system", "information", "data"
            ]
            text_parts.extend(np.random.choice(common_words, size=5))
            
            # Create document
            doc = {
                "id": f"synthetic_{i}",
                "title": f"Document {i}",
                "content": " ".join(text_parts),
                "categories": doc_categories,
                "filename": f"doc_{i}_{doc_categories[0]}.pdf"
            }
            
            synthetic_docs.append(doc)
        
        return synthetic_docs
    
    async def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            "pipeline": self.pipeline,
            "categories": self.categories,
            "training_stats": self.training_stats,
            "model_type": self.model_type,
            "hyperparameters": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "min_df": self.min_df,
                "max_df": self.max_df
            }
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save using joblib for sklearn models
        await asyncio.get_event_loop().run_in_executor(
            None, joblib.dump, model_data, filepath
        )
    
    async def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = await asyncio.get_event_loop().run_in_executor(
            None, joblib.load, filepath
        )
        
        self.pipeline = model_data["pipeline"]
        self.categories = model_data["categories"]
        self.training_stats = model_data["training_stats"]
        self.model_type = model_data.get("model_type", "logistic_regression")
        
        hyperparams = model_data.get("hyperparameters", {})
        self.max_features = hyperparams.get("max_features", self.max_features)
        self.ngram_range = hyperparams.get("ngram_range", self.ngram_range)
        self.min_df = hyperparams.get("min_df", self.min_df)
        self.max_df = hyperparams.get("max_df", self.max_df)
        
        self.is_trained = True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if not self.is_trained:
            return {"status": "not_trained", "message": "Model has not been trained"}
        
        return {
            "status": "trained",
            "model_type": self.model_type,
            "categories": self.categories,
            "training_stats": self.training_stats,
            "hyperparameters": {
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "min_df": self.min_df,
                "max_df": self.max_df
            }
        }
    
    async def evaluate_model(self, test_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model performance on test documents."""
        if not self.is_trained:
            raise ValueError("Model is not trained")
        
        if len(test_documents) == 0:
            raise ValueError("No test documents provided")
        
        # Prepare test data
        texts = []
        true_labels = []
        
        for doc in test_documents:
            text = self._extract_text_features(doc)
            categories = doc.get('categories', [])
            
            if text and categories:
                texts.append(text)
                true_labels.append(categories)
        
        if len(texts) == 0:
            raise ValueError("No valid test documents")
        
        # Convert to binary format
        y_true = self._labels_to_binary(true_labels, self.categories)
        
        # Get predictions
        y_pred = self.pipeline.predict(texts)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        return {
            "accuracy": float(accuracy),
            "num_test_docs": len(texts),
            "meets_target": accuracy >= 0.85,
            "evaluated_at": datetime.utcnow().isoformat()
        }