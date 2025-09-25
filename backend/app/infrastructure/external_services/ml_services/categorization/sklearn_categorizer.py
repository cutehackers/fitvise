"""Scikit-learn based document categorization service (improved with ColumnTransformer)."""

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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.multiclass import OneVsRestClassifier


class SklearnDocumentCategorizer:
    """ML categorizer - TF-IDF + LogisticRegression/NaiveBayes targeting >= 85% F1.
    Multi-label classification with confidence scoring, synthetic data generation & model persistence.
    """

    def __init__(
        self,
        model_type: str = "logistic_regression",
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.8,
        model_save_path: Optional[str] = None,
    ):
        self.model_type = model_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.model_save_path = model_save_path or "models/document_categorizer.pkl"

        self.pipeline = None
        self.categories = []
        self.is_trained = False
        self.training_stats = {}

        self.default_categories = [
            "financial",
            "legal",
            "hr",
            "technical",
            "marketing",
            "research",
            "policy",
            "manual",
            "report",
            "correspondence",
        ]

        self.category_keywords = {
            "financial": [
                "budget",
                "revenue",
                "profit",
                "loss",
                "expense",
                "invoice",
                "financial",
                "accounting",
                "balance",
                "statement",
                "audit",
            ],
            "legal": [
                "contract",
                "agreement",
                "legal",
                "law",
                "compliance",
                "regulation",
                "terms",
                "conditions",
                "liability",
                "rights",
                "clause",
            ],
            "hr": [
                "employee",
                "hr",
                "human resources",
                "personnel",
                "hiring",
                "training",
                "performance",
                "benefits",
                "salary",
                "policy",
                "onboarding",
            ],
            "technical": [
                "technical",
                "specification",
                "documentation",
                "api",
                "code",
                "system",
                "architecture",
                "design",
                "implementation",
                "development",
                "software",
            ],
            "marketing": [
                "marketing",
                "campaign",
                "promotion",
                "brand",
                "customer",
                "market",
                "advertising",
                "social media",
                "content",
                "strategy",
                "audience",
            ],
            "research": [
                "research",
                "study",
                "analysis",
                "data",
                "findings",
                "methodology",
                "hypothesis",
                "conclusion",
                "experiment",
                "survey",
                "statistics",
            ],
            "policy": [
                "policy",
                "procedure",
                "guideline",
                "standard",
                "rule",
                "protocol",
                "governance",
                "framework",
                "compliance",
                "requirement",
                "process",
            ],
            "manual": [
                "manual",
                "guide",
                "instructions",
                "handbook",
                "tutorial",
                "how-to",
                "step-by-step",
                "user guide",
                "reference",
                "documentation",
                "help",
            ],
            "report": [
                "report",
                "summary",
                "overview",
                "status",
                "progress",
                "quarterly",
                "annual",
                "monthly",
                "executive",
                "dashboard",
                "metrics",
            ],
            "correspondence": [
                "email",
                "letter",
                "memo",
                "message",
                "communication",
                "notice",
                "announcement",
                "update",
                "notification",
                "correspondence",
                "reply",
            ],
        }

    def _create_pipeline(self) -> Pipeline:
        vectorizer_params = dict(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
        )

        # Column-wise vectorizers with weights
        preprocessor = ColumnTransformer(
            transformers=[
                ("title", TfidfVectorizer(**vectorizer_params), "title"),
                ("description", TfidfVectorizer(**vectorizer_params), "description"),
                ("filename", TfidfVectorizer(**vectorizer_params), "filename"),
                ("keywords", TfidfVectorizer(**vectorizer_params), "keywords"),
                ("content", TfidfVectorizer(**vectorizer_params), "content"),
            ],
            remainder="drop",
        )

        if self.model_type == "naive_bayes":
            classifier = MultinomialNB(alpha=0.1)
        else:
            classifier = OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1))

        return Pipeline([("vectorizer", preprocessor), ("classifier", classifier)])

    async def train_model(
        self, documents: List[Dict[str, Any]], test_size: float = 0.2, cross_validation_folds: int = 5
    ) -> Dict[str, Any]:
        if len(documents) < 10:
            raise ValueError("Need at least 10 documents for training")

        # Prepare data into pandas DataFrame for ColumnTransformer
        data = []
        labels = []
        for doc in documents:
            categories = doc.get("categories", [])
            if not categories:
                continue
            row = {
                "title": doc.get("title", ""),
                "description": doc.get("description", ""),
                "filename": doc.get("filename", ""),
                "keywords": " ".join(doc.get("keywords", [])),
                "content": doc.get("content", ""),
            }
            if any(row.values()):
                data.append(row)
                labels.append(categories)

        if len(data) < 10:
            raise ValueError("Not enough valid training documents")

        df = pd.DataFrame(data)
        self.categories = list(set([cat for cats in labels for cat in cats]))
        y_binary = self._labels_to_binary(labels, self.categories)

        X_train, X_test, y_train, y_test = train_test_split(df, y_binary, test_size=test_size, random_state=42)

        self.pipeline = self._create_pipeline()
        start_time = datetime.now(datetime.timezone.utc)
        self.pipeline.fit(X_train, y_train)
        training_time = (datetime.now(datetime.timezone.utc) - start_time).total_seconds()

        y_pred = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_micro = f1_score(y_test, y_pred, average="micro")
        f1_macro = f1_score(y_test, y_pred, average="macro")

        if cross_validation_folds > 1:
            cv_scores = cross_val_score(
                self.pipeline, df, y_binary, cv=cross_validation_folds, scoring="f1_micro", n_jobs=-1
            )
        else:
            cv_scores = [f1_micro]

        self.is_trained = True
        self.training_stats = {
            "training_time_seconds": training_time,
            "num_documents": len(df),
            "num_categories": len(self.categories),
            "test_accuracy": acc,
            "test_f1_micro": f1_micro,
            "test_f1_macro": f1_macro,
            "cv_mean_f1_micro": np.mean(cv_scores),
            "cv_std_f1_micro": np.std(cv_scores),
            "meets_target": f1_micro >= 0.85,
            "trained_at": datetime.now(datetime.timezone.utc).isoformat(),
            "categories": self.categories,
        }

        if self.model_save_path:
            await self.save_model(self.model_save_path)

        return self.training_stats

    async def predict_categories(self, text: str, top_k: int = 3, min_confidence: float = 0.1) -> List[Dict[str, Any]]:
        if not self.is_trained:
            raise ValueError("Model is not trained. Call train_model() first.")
        if not text.strip():
            return []

        df = pd.DataFrame([{"title": "", "description": "", "filename": "", "keywords": "", "content": text}])
        proba_list = self.pipeline.predict_proba(df)
        probabilities = np.array([p[0][1] for p in proba_list])
        predictions = self.pipeline.predict(df)[0]

        results = []
        for category, prob, pred in zip(self.categories, probabilities, predictions):
            if pred == 1 and prob >= min_confidence:
                results.append({"category": category, "confidence": float(prob), "predicted": bool(pred)})

        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results[:top_k]

    def _labels_to_binary(self, labels: List[List[str]], all_categories: List[str]) -> np.ndarray:
        binary_labels = np.zeros((len(labels), len(all_categories)))
        for i, doc_labels in enumerate(labels):
            for label in doc_labels:
                if label in all_categories:
                    j = all_categories.index(label)
                    binary_labels[i, j] = 1
        return binary_labels

    async def save_model(self, filepath: str) -> None:
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
                "max_df": self.max_df,
            },
            "version": "1.1",
            "sklearn_version": "1.5",
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        await asyncio.get_event_loop().run_in_executor(None, joblib.dump, model_data, filepath)

    async def load_model(self, filepath: str) -> None:
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        model_data = await asyncio.get_event_loop().run_in_executor(None, joblib.load, filepath)
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
                "max_df": self.max_df,
            },
        }

    async def evaluate_model(self, test_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Model is not trained")
        if len(test_documents) == 0:
            raise ValueError("No test documents provided")

        data, true_labels = [], []
        for doc in test_documents:
            categories = doc.get("categories", [])
            row = {
                "title": doc.get("title", ""),
                "description": doc.get("description", ""),
                "filename": doc.get("filename", ""),
                "keywords": " ".join(doc.get("keywords", [])),
                "content": doc.get("content", ""),
            }
            if any(row.values()) and categories:
                data.append(row)
                true_labels.append(categories)

        if len(data) == 0:
            raise ValueError("No valid test documents")

        df = pd.DataFrame(data)
        y_true = self._labels_to_binary(true_labels, self.categories)
        y_pred = self.pipeline.predict(df)

        acc = accuracy_score(y_true, y_pred)
        f1_micro = f1_score(y_true, y_pred, average="micro")
        f1_macro = f1_score(y_true, y_pred, average="macro")

        return {
            "accuracy": float(acc),
            "f1_micro": float(f1_micro),
            "f1_macro": float(f1_macro),
            "num_test_docs": len(df),
            "meets_target": f1_micro >= 0.85,
            "evaluated_at": datetime.utcnow().isoformat(),
        }
