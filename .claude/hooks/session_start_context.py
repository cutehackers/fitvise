#!/usr/bin/env python3
import json
import os
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime

# Security validation rules
SECURITY_RULES = [
    (r"\b(rm\s+-rf|sudo\s+rm)", "Dangerous deletion commands detected"),
    (r"\b(chmod\s+777|chown\s+root)", "Unsafe permission changes detected"),
    (r"\b(curl\s+.*\|\s*sh|wget\s+.*\|\s*sh)", "Piping downloads to shell detected"),
    (
        r"\b(export\s+.*PASSWORD|export\s+.*TOKEN)",
        "Exposed credentials in command detected",
    ),
]

# Prompt validation rules
PROMPT_VALIDATION_RULES = [
    (
        r"\bgrep\b(?!.*\|)",
        "Consider using 'rg' (ripgrep) instead of 'grep' for better performance",
    ),
    (
        r"\bfind\s+.*-name",
        "Consider using 'rg --files -g pattern' instead of 'find -name' for better performance",
    ),
]


def read_claude_md() -> str:
    """Read and return content from CLAUDE.md file if it exists."""
    claude_md_paths = [
        "CLAUDE.md",
        "INSTRUCTIONS.md",
    ]

    for path in claude_md_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        return f"## Project Context from {path}\n\n{content}\n"
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}", file=sys.stderr)

    return ""


def get_project_metadata() -> str:
    """Gather project metadata like framework, dependencies, git status."""
    metadata = []

    # Detect project type and framework
    if os.path.exists("pubspec.yaml"):
        metadata.append("Framework: Flutter/Dart")
        try:
            with open("pubspec.yaml", "r") as f:
                content = f.read()
                # Extract key dependencies
                deps = re.findall(
                    r"^\s+([a-zA-Z][a-zA-Z0-9_]*):.*$", content, re.MULTILINE
                )
                if deps:
                    key_deps = [
                        dep
                        for dep in deps[:5]
                        if dep not in ["flutter", "cupertino_icons"]
                    ]
                    if key_deps:
                        metadata.append(f"Key Dependencies: {', '.join(key_deps)}")
        except Exception:
            pass
    elif os.path.exists("package.json"):
        metadata.append("Framework: Node.js")
        try:
            with open("package.json", "r") as f:
                import json as json_lib

                pkg = json_lib.load(f)
                deps = list(pkg.get("dependencies", {}).keys())[:5]
                if deps:
                    metadata.append(f"Key Dependencies: {', '.join(deps)}")
        except Exception:
            pass
    elif os.path.exists("requirements.txt"):
        metadata.append("Framework: Python")
        try:
            with open("requirements.txt", "r") as f:
                deps = [
                    line.split("==")[0].split(">=")[0].strip()
                    for line in f.readlines()[:5]
                    if line.strip() and not line.startswith("#")
                ]
                if deps:
                    metadata.append(f"Key Dependencies: {', '.join(deps)}")
        except Exception:
            pass
    elif os.path.exists("Cargo.toml"):
        metadata.append("Framework: Rust")
    elif os.path.exists("go.mod"):
        metadata.append("Framework: Go")

    # Git information
    if os.path.exists(".git"):
        try:
            # Current branch
            result = subprocess.run(
                ["git", "branch", "--show-current"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                metadata.append(f"Git Branch: {result.stdout.strip()}")

            # Uncommitted changes
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                changes = len(
                    [line for line in result.stdout.strip().split("\n") if line.strip()]
                )
                if changes > 0:
                    metadata.append(f"Uncommitted Changes: {changes} files")
                else:
                    metadata.append("Working Directory: Clean")

            # Recent commits
            result = subprocess.run(
                ["git", "log", "--oneline", "-3"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                commits = result.stdout.strip().split("\n")
                metadata.append(f"Recent Commits: {len(commits)} commits")

        except Exception:
            metadata.append("Git: Available")

    # Recent files
    try:
        extensions = [".dart", ".js", ".ts", ".py", ".rs", ".go", ".java", ".cpp", ".c"]
        recent_files = []

        for ext in extensions:
            try:
                result = subprocess.run(
                    [
                        "find",
                        ".",
                        "-name",
                        f"*{ext}",
                        "-type",
                        "f",
                        "!",
                        "-path",
                        "./node_modules/*",
                        "!",
                        "-path",
                        "./.git/*",
                        "!",
                        "-path",
                        "./build/*",
                        "-printf",
                        "%T@ %p\n",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    files = [
                        (float(line.split()[0]), line.split()[1])
                        for line in result.stdout.strip().split("\n")
                        if line.strip()
                    ]
                    files.sort(reverse=True)
                    recent_files.extend([f[1] for f in files[:3]])
            except Exception:
                continue

        if recent_files:
            metadata.append(f"Recent Files: {', '.join(recent_files[:5])}")

    except Exception:
        pass

    if metadata:
        return (
            "## Project Metadata\n\n"
            + "\n".join(f"- {item}" for item in metadata)
            + "\n"
        )
    return ""


def validate_security(prompt: str) -> list[str]:
    """Check prompt for security issues."""
    issues = []
    for pattern, message in SECURITY_RULES:
        if re.search(pattern, prompt, re.IGNORECASE):
            issues.append(message)
    return issues


def validate_prompt(prompt: str) -> list[str]:
    """Check prompt for best practices."""
    suggestions = []
    for pattern, message in PROMPT_VALIDATION_RULES:
        if re.search(pattern, prompt):
            suggestions.append(message)
    return suggestions


def main():
    try:
        # Read input from stdin
        input_data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract data from input
    prompt = input_data.get("prompt", "")
    session_id = input_data.get("session_id", "unknown")

    if not prompt:
        sys.exit(0)

    # Security validation
    security_issues = validate_security(prompt)
    if security_issues:
        error_msg = "Security validation failed:\n" + "\n".join(
            f"• {issue}" for issue in security_issues
        )

        # Return JSON to block the prompt
        result = {
            "decision": "block",
            "reason": error_msg,
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": "",
            },
        }
        print(json.dumps(result))
        sys.exit(2)

    # Build additional context
    additional_context = ""

    # Add CLAUDE.md content
    claude_md_content = read_claude_md()
    if claude_md_content:
        additional_context += claude_md_content + "\n"

    # Add project metadata
    project_metadata = get_project_metadata()
    if project_metadata:
        additional_context += project_metadata + "\n"

    # Add timestamp
    additional_context += f"## Session Info\n\n- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    additional_context += f"- Session ID: {session_id}\n\n"

    # Check for prompt suggestions (non-blocking)
    suggestions = validate_prompt(prompt)
    if suggestions:
        suggestion_text = (
            "## Suggestions\n\n"
            + "\n".join(f"- {suggestion}" for suggestion in suggestions)
            + "\n\n"
        )
        additional_context += suggestion_text

    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context.strip(),
        }
    }

    # Return JSON with additional context
    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
