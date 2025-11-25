"""Prompt service for managing chat prompts (Phase 3 refactoring).

This module defines the PromptService domain service that manages prompt templates,
system prompts, and prompt building for chat conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from app.domain.entities.retrieval_context import RetrievalContext
from app.domain.llm.entities.message import Message
from app.domain.entities.message_role import MessageRole


@dataclass
class PromptTemplate:
    """Domain entity representing a prompt template.

    Attributes:
        template_id: Unique identifier for the template
        name: Human-readable name for the template
        description: Template description
        template_text: Template text with placeholders
        placeholders: List of placeholder names in the template
        metadata: Additional template metadata
        created_at: When the template was created
    """

    template_id: UUID
    name: str
    description: str
    template_text: str
    placeholders: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default="")

    def __post_init__(self) -> None:
        """Initialize placeholders and created_at."""
        if not self.template_text:
            raise ValueError("Template text cannot be empty")

        # Extract placeholders from template text
        self.placeholders = self._extract_placeholders()

        # Set created_at if not provided
        if not self.created_at:
            from datetime import datetime
            self.created_at = datetime.utcnow().isoformat()

    def _extract_placeholders(self) -> List[str]:
        """Extract placeholder names from template text.

        Returns:
            List of placeholder names
        """
        import re
        # Find {placeholder} patterns
        pattern = r'\{([^}]+)\}'
        matches = re.findall(pattern, self.template_text)
        return list(set(matches))

    def render(self, variables: Dict[str, Any]) -> str:
        """Render the template with provided variables.

        Args:
            variables: Dictionary of variable values

        Returns:
            Rendered template text

        Raises:
            ValueError: If required placeholders are missing
        """
        # Check for missing placeholders
        missing_placeholders = [
            placeholder for placeholder in self.placeholders
            if placeholder not in variables
        ]

        if missing_placeholders:
            raise ValueError(f"Missing required placeholders: {missing_placeholders}")

        # Render template
        try:
            return self.template_text.format(**variables)
        except KeyError as e:
            raise ValueError(f"Missing placeholder: {str(e)}") from e
        except ValueError as e:
            raise ValueError(f"Template rendering error: {str(e)}") from e

    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate variables against template requirements.

        Args:
            variables: Variables to validate

        Returns:
            List of missing variable names
        """
        return [
            placeholder for placeholder in self.placeholders
            if placeholder not in variables
        ]


@dataclass
class PromptConfig:
    """Configuration for prompt building operations.

    Attributes:
        max_context_length: Maximum length of context in prompts
        include_citations: Whether to include citations in prompts
        citation_format: Format for citations (e.g., "[1]", "[Source: 1]")
        system_prompt_style: Style for system prompts (concise, detailed, friendly)
        enable_context_compression: Whether to compress context when needed
    """

    max_context_length: int = 3000
    include_citations: bool = True
    citation_format: str = "[{index}]"
    system_prompt_style: str = "concise"
    enable_context_compression: bool = True


class PromptService:
    """Domain service for managing chat prompts.

    Provides business logic for prompt template management, prompt building,
    and context integration for chat conversations.

    Responsibilities:
    - Create and manage prompt templates
    - Build prompts with context and conversation history
    - Handle system prompt generation and customization
    - Manage prompt formatting and optimization
    - Provide prompt validation and error handling

    Examples:
        >>> service = PromptService()
        >>> prompt = service.build_rag_prompt(query, context, history)
        >>> template = service.create_template("fitness_assistant", template_text)
        >>> rendered = service.render_template(template_id, variables)
    """

    def __init__(self, config: Optional[PromptConfig] = None) -> None:
        """Initialize prompt service.

        Args:
            config: Optional prompt building configuration
        """
        self._config = config or PromptConfig()
        self._templates: Dict[str, PromptTemplate] = {}
        self._initialize_default_templates()

    def _initialize_default_templates(self) -> None:
        """Initialize default prompt templates."""
        # RAG chat template
        rag_template = PromptTemplate(
            template_id=UUID(),
            name="rag_chat",
            description="RAG-enabled chat template with context and citations",
            template_text=(
                "You are a helpful fitness assistant. Answer questions using the provided context.\n\n"
                "IMPORTANT: When using information from the context, cite your sources using [1], [2], etc.\n\n"
                "Context:\n{context}\n\n"
                "If the context doesn't contain relevant information, say so clearly.\n\n"
                "Conversation History:\n{history}\n\n"
                "Human: {query}\n\n"
                "Assistant:"
            ),
        )
        self._templates["rag_chat"] = rag_template

        # Basic chat template
        basic_template = PromptTemplate(
            template_id=UUID(),
            name="basic_chat",
            description="Basic chat template without context",
            template_text=(
                "You are a helpful fitness assistant.\n\n"
                "Conversation History:\n{history}\n\n"
                "Human: {query}\n\n"
                "Assistant:"
            ),
        )
        self._templates["basic_chat"] = basic_template

        # System prompt template
        system_template = PromptTemplate(
            template_id=UUID(),
            name="system_prompt",
            description="System prompt template",
            template_text=(
                "You are a helpful AI fitness assistant. "
                "Use the provided context to answer the user's question thoroughly and accurately. "
                "If the context contains relevant information, base your answer on it. "
                "Always cite your sources when using information from the context.\n\n"
                "Context:\n{context}"
            ),
        )
        self._templates["system_prompt"] = system_template

    def create_template(
        self,
        name: str,
        template_text: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> PromptTemplate:
        """Create a new prompt template.

        Args:
            name: Template name
            template_text: Template text with placeholders
            description: Template description
            metadata: Optional template metadata

        Returns:
            Created prompt template

        Raises:
            ValueError: If template name already exists or text is invalid
        """
        if name in self._templates:
            raise ValueError(f"Template '{name}' already exists")

        template = PromptTemplate(
            template_id=UUID(),
            name=name,
            description=description,
            template_text=template_text,
            metadata=metadata or {},
        )

        self._templates[name] = template
        return template

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name.

        Args:
            name: Template name

        Returns:
            Prompt template if found, None otherwise
        """
        return self._templates.get(name)

    def render_template(
        self,
        template_name: str,
        variables: Dict[str, Any],
    ) -> str:
        """Render a prompt template with variables.

        Args:
            template_name: Name of template to render
            variables: Variables to substitute in template

        Returns:
            Rendered prompt text

        Raises:
            ValueError: If template not found or variables invalid
        """
        template = self.get_template(template_name)
        if template is None:
            raise ValueError(f"Template '{template_name}' not found")

        return template.render(variables)

    def build_rag_prompt(
        self,
        query: str,
        context: RetrievalContext,
        history: Optional[List[Message]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build a RAG-enabled prompt.

        Args:
            query: User query
            context: Retrieval context with documents
            history: Optional conversation history
            system_prompt: Optional custom system prompt

        Returns:
            Built prompt text
        """
        # Format context
        formatted_context = self._format_context(context)

        # Format history
        formatted_history = self._format_history(history or [])

        # Use custom system prompt or default
        final_system_prompt = system_prompt or self._get_default_system_prompt()

        # Build variables for template
        variables = {
            "context": formatted_context,
            "history": formatted_history,
            "query": query,
            "system_prompt": final_system_prompt,
        }

        try:
            return self.render_template("rag_chat", variables)
        except ValueError:
            # Fallback to manual prompt building
            return self._build_fallback_rag_prompt(query, formatted_context, formatted_history)

    def build_basic_prompt(
        self,
        query: str,
        history: Optional[List[Message]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build a basic chat prompt without context.

        Args:
            query: User query
            history: Optional conversation history
            system_prompt: Optional custom system prompt

        Returns:
            Built prompt text
        """
        # Format history
        formatted_history = self._format_history(history or [])

        # Use custom system prompt or default
        final_system_prompt = system_prompt or self._get_default_system_prompt()

        variables = {
            "history": formatted_history,
            "query": query,
            "system_prompt": final_system_prompt,
        }

        try:
            return self.render_template("basic_chat", variables)
        except ValueError:
            # Fallback to manual prompt building
            return self._build_fallback_basic_prompt(query, formatted_history)

    def _format_context(self, context: RetrievalContext) -> str:
        """Format retrieval context for prompt.

        Args:
            context: Retrieval context to format

        Returns:
            Formatted context text
        """
        if not context.has_context() or not context.context_text:
            return "No relevant context found."

        formatted_parts = []
        for i, ref in enumerate(context.document_references, 1):
            doc_title = ref.get_document_title()
            doc_type = ref.get_document_type()
            content = ref.get_content_excerpt(500)

            citation = self._config.citation_format.format(index=i)
            formatted_parts.append(f"{citation} [{doc_type}: {doc_title}]\n{content}")

        return "\n\n".join(formatted_parts)

    def _format_history(self, history: List[Message]) -> str:
        """Format conversation history for prompt.

        Args:
            history: List of messages to format

        Returns:
            Formatted history text
        """
        if not history:
            return "No previous conversation."

        formatted_messages = []
        for message in history:
            role_name = message.role.value.title()
            content = message.content
            formatted_messages.append(f"{role_name}: {content}")

        return "\n".join(formatted_messages)

    def _get_default_system_prompt(self) -> str:
        """Get default system prompt based on configuration.

        Returns:
            Default system prompt text
        """
        style = self._config.system_prompt_style

        if style == "concise":
            return "You are a helpful fitness assistant."
        elif style == "detailed":
            return (
                "You are a knowledgeable fitness assistant with expertise in exercise science, "
                "nutrition, and wellness. Provide accurate, helpful responses based on reliable information."
            )
        elif style == "friendly":
            return (
                "You're a friendly fitness coach here to help people achieve their health and wellness goals. "
                "Be encouraging, supportive, and provide practical advice."
            )
        else:
            return "You are a helpful fitness assistant."

    def _build_fallback_rag_prompt(
        self,
        query: str,
        context: str,
        history: str,
    ) -> str:
        """Build fallback RAG prompt if template rendering fails.

        Args:
            query: User query
            context: Formatted context
            history: Formatted history

        Returns:
            Fallback prompt text
        """
        return (
            f"{self._get_default_system_prompt()}\n\n"
            f"Context:\n{context}\n\n"
            f"Conversation History:\n{history}\n\n"
            f"Human: {query}\n\n"
            f"Assistant:"
        )

    def _build_fallback_basic_prompt(self, query: str, history: str) -> str:
        """Build fallback basic prompt if template rendering fails.

        Args:
            query: User query
            history: Formatted history

        Returns:
            Fallback prompt text
        """
        return (
            f"{self._get_default_system_prompt()}\n\n"
            f"Conversation History:\n{history}\n\n"
            f"Human: {query}\n\n"
            f"Assistant:"
        )

    def validate_prompt_length(self, prompt: str, max_length: Optional[int] = None) -> Tuple[bool, int]:
        """Validate prompt length against maximum.

        Args:
            prompt: Prompt text to validate
            max_length: Maximum allowed length (uses config if not provided)

        Returns:
            Tuple of (is_valid, actual_length)
        """
        max_len = max_length or self._config.max_context_length
        actual_length = len(prompt)
        return actual_length <= max_len, actual_length

    def compress_prompt(self, prompt: str, target_length: int) -> str:
        """Compress prompt to target length.

        Args:
            prompt: Prompt text to compress
            target_length: Target maximum length

        Returns:
            Compressed prompt text
        """
        if len(prompt) <= target_length:
            return prompt

        # Simple compression: truncate from middle, keep start and end
        if target_length < 100:
            return prompt[:target_length]

        keep_start = target_length // 2 - 50
        keep_end = target_length // 2 - 50

        return f"{prompt[:keep_start]}...[truncated]...{prompt[-keep_end:]}"

    def get_template_list(self) -> List[Dict[str, Any]]:
        """Get list of all available templates.

        Returns:
            List of template information
        """
        return [
            {
                "name": template.name,
                "description": template.description,
                "placeholders": template.placeholders,
                "created_at": template.created_at,
            }
            for template in self._templates.values()
        ]

    def delete_template(self, name: str) -> bool:
        """Delete a prompt template.

        Args:
            name: Template name to delete

        Returns:
            True if template was deleted, False if not found
        """
        if name in self._templates:
            del self._templates[name]
            return True
        return False

    def get_prompt_statistics(self) -> Dict[str, Any]:
        """Get prompt service statistics.

        Returns:
            Dictionary with service statistics
        """
        return {
            "total_templates": len(self._templates),
            "config": {
                "max_context_length": self._config.max_context_length,
                "include_citations": self._config.include_citations,
                "citation_format": self._config.citation_format,
                "system_prompt_style": self._config.system_prompt_style,
                "enable_context_compression": self._config.enable_context_compression,
            },
            "available_templates": self.get_template_list(),
        }