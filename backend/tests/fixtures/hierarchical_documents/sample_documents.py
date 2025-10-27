"""Sample hierarchical documents for testing Task 2.1.3."""

# Policy document with clear 3-level hierarchy
POLICY_DOCUMENT = """# Company Policy Manual

## Section 1: Code of Conduct

### 1.1 Professional Behavior
Employees must maintain professional standards at all times. This includes appropriate workplace attire, punctuality, and respectful communication with colleagues and clients.

### 1.2 Workplace Ethics
Ethical conduct is the foundation of our organization. Employees are expected to demonstrate integrity, honesty, and transparency in all business dealings.

### 1.3 Conflict of Interest
Employees must disclose any potential conflicts of interest. This includes financial interests, family relationships, or outside business activities that may interfere with job responsibilities.

## Section 2: Leave Policies

### 2.1 Annual Leave
Employees are entitled to 20 days of paid annual leave per year. Leave requests must be submitted at least two weeks in advance for approval by the direct supervisor.

### 2.2 Sick Leave
Employees receive 10 days of paid sick leave annually. Medical certificates are required for absences exceeding three consecutive days.

### 2.3 Family Leave
Family and medical leave is available for eligible employees. This includes parental leave, care for seriously ill family members, and adoption leave.

## Section 3: Performance Management

### 3.1 Performance Reviews
Annual performance reviews are conducted for all employees. Reviews assess job performance, goal achievement, and professional development needs.

### 3.2 Goal Setting
Employees work with supervisors to establish clear, measurable goals. Goals are reviewed quarterly and adjusted as needed to align with business objectives.

### 3.3 Professional Development
The company supports employee growth through training programs, conferences, and educational opportunities. Employees may request professional development funding.
"""

# Shorter policy document for testing
SHORT_POLICY = """# Employee Handbook

## Introduction
Welcome to the company. This handbook outlines key policies and procedures.

## Work Hours
Standard work hours are 9 AM to 5 PM, Monday through Friday.

## Dress Code
Business casual attire is required for office employees.
"""

# Technical manual with deep nesting
TECHNICAL_MANUAL = """# System Architecture Guide

## Chapter 1: Overview

### 1.1 System Components
The system consists of multiple interconnected components working together to provide seamless functionality.

#### 1.1.1 Frontend Layer
The frontend layer handles user interactions and displays information through a responsive web interface.

#### 1.1.2 Backend Layer
The backend layer processes business logic and manages data persistence.

### 1.2 Data Flow
Data flows through the system following established patterns and protocols.

#### 1.2.1 Request Processing
Incoming requests are validated, processed, and routed to appropriate handlers.

#### 1.2.2 Response Generation
Responses are formatted according to API specifications and returned to clients.

## Chapter 2: Implementation Details

### 2.1 Technology Stack
The system is built using modern, industry-standard technologies.

#### 2.1.1 Programming Languages
Primary development uses Python for backend services and TypeScript for frontend applications.

#### 2.1.2 Frameworks and Libraries
We utilize FastAPI for REST APIs, React for UI components, and PostgreSQL for data storage.
"""

# Markdown document with various heading levels
MARKDOWN_DOC = """# Main Title

## Section A

### Subsection A.1
Content for subsection A.1 with detailed information.

### Subsection A.2
Content for subsection A.2 with additional details.

## Section B

### Subsection B.1
Important information in subsection B.1.

#### Deep Level B.1.1
Very specific details at the deepest level.

### Subsection B.2
More content for subsection B.2.
"""

# Simple flat document (single level)
FLAT_DOCUMENT = """This is a simple document without any hierarchical structure.
It contains multiple paragraphs of text but no headings or sections.

This second paragraph continues the narrative without introducing any structural elements.

The third paragraph concludes the document with final thoughts and summary information.
"""

# Empty and edge case documents
EMPTY_DOCUMENT = ""

WHITESPACE_ONLY = "   \n\n\t  \n   "

SINGLE_SENTENCE = "This is a single sentence document."

# Document with only headings (no content)
HEADINGS_ONLY = """# Title
## Section 1
### Subsection 1.1
## Section 2
### Subsection 2.1
### Subsection 2.2
"""

# All test documents organized by category
ALL_DOCUMENTS = {
    "policy": POLICY_DOCUMENT,
    "short_policy": SHORT_POLICY,
    "technical_manual": TECHNICAL_MANUAL,
    "markdown": MARKDOWN_DOC,
    "flat": FLAT_DOCUMENT,
    "empty": EMPTY_DOCUMENT,
    "whitespace": WHITESPACE_ONLY,
    "single_sentence": SINGLE_SENTENCE,
    "headings_only": HEADINGS_ONLY,
}

# Expected hierarchy depths for each document type
EXPECTED_DEPTHS = {
    "policy": 3,  # Document > Section > Subsection
    "short_policy": 2,  # Document > Section
    "technical_manual": 4,  # Chapter > Section > Subsection > Deep level
    "markdown": 3,  # Title > Section > Subsection
    "flat": 1,  # Single level
    "single_sentence": 1,  # Single level
}
