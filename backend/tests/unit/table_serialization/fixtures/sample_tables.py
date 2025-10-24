"""Sample tables from various domains for testing (Task 2.1.2).

Provides realistic test data representing:
- Financial reports (complex multi-level headers)
- Medical lab results (scientific data)
- API documentation (technical specifications)
- Chemistry data (scientific measurements)
- General business data
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np


def get_financial_table() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a financial report table with multi-level headers.

    Represents a quarterly financial summary with actual vs budget columns.
    Tests: Multi-level headers, numeric data, percentage formatting.
    """
    # Create multi-level columns
    columns = pd.MultiIndex.from_tuples([
        ("", "Metric"),
        ("Q1 2024", "Actual"),
        ("Q1 2024", "Budget"),
        ("Q2 2024", "Actual"),
        ("Q2 2024", "Budget"),
    ])

    data = [
        ["Revenue ($M)", 125.5, 120.0, 138.7, 135.0],
        ["COGS ($M)", 65.2, 62.0, 70.1, 68.0],
        ["Gross Profit ($M)", 60.3, 58.0, 68.6, 67.0],
        ["Operating Expenses ($M)", 28.5, 30.0, 29.2, 31.0],
        ["Net Income ($M)", 31.8, 28.0, 39.4, 36.0],
    ]

    df = pd.DataFrame(data, columns=columns)

    metadata = {
        "file_name": "financial_report_q2_2024.pdf",
        "page": 15,
        "caption": "Quarterly Financial Performance Summary",
        "processor": "docling",
    }

    return df, metadata


def get_medical_lab_table() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a medical lab results table.

    Represents patient lab test results.
    Tests: Mixed data types, reference ranges, status indicators.
    """
    data = {
        "Test Name": ["Hemoglobin", "White Blood Cell", "Platelets", "Glucose", "Cholesterol"],
        "Patient ID": ["P-12345", "P-12345", "P-12345", "P-12345", "P-12345"],
        "Result": [14.2, 7.8, 180, 95, 185],
        "Reference Range": ["13.5-17.5 g/dL", "4.5-11.0 K/μL", "150-400 K/μL", "70-100 mg/dL", "<200 mg/dL"],
        "Status": ["Normal", "Normal", "Normal", "Normal", "Normal"],
    }

    df = pd.DataFrame(data)

    metadata = {
        "file_name": "lab_results_2024_10_24.pdf",
        "page": 1,
        "caption": "Complete Blood Count and Metabolic Panel",
        "processor": "docling",
    }

    return df, metadata


def get_api_documentation_table() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create an API endpoints documentation table.

    Represents REST API endpoint specifications.
    Tests: String data, technical terminology, structured documentation.
    """
    data = {
        "Endpoint": ["/api/users", "/api/users/:id", "/api/users", "/api/auth/login", "/api/data/export"],
        "Method": ["GET", "GET", "POST", "POST", "GET"],
        "Auth Required": ["Yes", "Yes", "Yes", "No", "Yes"],
        "Rate Limit": ["100/min", "100/min", "10/min", "5/min", "20/min"],
        "Response Format": ["JSON", "JSON", "JSON", "JSON", "CSV/JSON"],
    }

    df = pd.DataFrame(data)

    metadata = {
        "file_name": "api_documentation_v2.md",
        "page": None,
        "caption": "API Endpoints Reference",
        "processor": "markdown",
    }

    return df, metadata


def get_chemistry_table() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a chemistry data table with scientific measurements.

    Represents chemical compound properties.
    Tests: Scientific notation, units, decimal precision.
    """
    data = {
        "Compound": ["Ethanol", "Acetone", "Benzene", "Toluene", "Methanol"],
        "Molecular Weight": [46.07, 58.08, 78.11, 92.14, 32.04],
        "Boiling Point (°C)": [78.4, 56.0, 80.1, 110.6, 64.7],
        "Density (g/cm³)": [0.789, 0.784, 0.876, 0.867, 0.792],
        "Solubility": ["Miscible", "Miscible", "Insoluble", "Insoluble", "Miscible"],
    }

    df = pd.DataFrame(data)

    metadata = {
        "file_name": "organic_compounds_properties.pdf",
        "page": 42,
        "caption": "Physical Properties of Common Organic Solvents",
        "processor": "docling",
    }

    return df, metadata


def get_business_table() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a general business data table.

    Represents sales performance by region.
    Tests: Simple structure, basic business metrics.
    """
    data = {
        "Region": ["North", "South", "East", "West", "Central"],
        "Sales ($K)": [450, 380, 520, 410, 390],
        "Customers": [125, 98, 142, 110, 105],
        "Growth (%)": [12.5, 8.3, 15.2, 10.1, 9.7],
        "Target Met": ["Yes", "No", "Yes", "Yes", "No"],
    }

    df = pd.DataFrame(data)

    metadata = {
        "file_name": "sales_report_october_2024.xlsx",
        "page": None,
        "caption": "Regional Sales Performance",
        "processor": "excel",
    }

    return df, metadata


def get_table_with_nulls() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a table with null/NaN values.

    Tests: Null handling, incomplete data, data quality issues.
    """
    data = {
        "Product": ["Widget A", "Widget B", "Widget C", "Widget D"],
        "Price": [19.99, np.nan, 34.50, 15.00],
        "Stock": [100, 50, np.nan, 200],
        "Category": ["Electronics", None, "Hardware", "Electronics"],
        "Supplier": ["Supplier X", "Supplier Y", "Supplier Z", np.nan],
    }

    df = pd.DataFrame(data)

    metadata = {
        "file_name": "inventory_incomplete.csv",
        "page": None,
        "caption": "Inventory with Missing Data",
        "processor": "csv",
    }

    return df, metadata


def get_table_with_merged_cells() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create a table simulating merged cells.

    Tests: Repeated values indicating merged regions, hierarchical grouping.
    """
    data = {
        "Category": ["Electronics", "Electronics", "Electronics", "Furniture", "Furniture", "Clothing", "Clothing"],
        "Subcategory": ["Phones", "Phones", "Laptops", "Office", "Home", "Men", "Women"],
        "Product": ["iPhone", "Samsung", "MacBook", "Desk", "Chair", "T-Shirt", "Dress"],
        "Price": [999, 849, 1299, 450, 200, 25, 60],
    }

    df = pd.DataFrame(data)

    metadata = {
        "file_name": "product_catalog.pdf",
        "page": 5,
        "caption": "Product Hierarchy and Pricing",
        "processor": "docling",
    }

    return df, metadata


# Registry of all sample tables
SAMPLE_TABLES: Dict[str, callable] = {
    "financial": get_financial_table,
    "medical": get_medical_lab_table,
    "api_docs": get_api_documentation_table,
    "chemistry": get_chemistry_table,
    "business": get_business_table,
    "with_nulls": get_table_with_nulls,
    "with_merged_cells": get_table_with_merged_cells,
}
