# #!/usr/bin/env python3
# import json
# import sys
# import re
# import datetime

# # Load input from stdin
# try:
#     input_data = json.load(sys.stdin)
# except json.JSONDecodeError as e:
#     print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
#     sys.exit(1)

# prompt = input_data.get("prompt", "")

# # Check for sensitive patterns
# sensitive_patterns = [
#     (r"(?i)\b(password|secret|key|token)\s*[:=]", "Prompt contains potential secrets"),
# ]

# for pattern, message in sensitive_patterns:
#     if re.search(pattern, prompt):
#         # Use JSON output to block with a specific reason
#         output = {
#             "decision": "block",
#             "reason": f"Security policy violation: {message}. Please rephrase your request without sensitive information."
#         }
#         print(json.dumps(output))
#         sys.exit(0)

# # Add current time to context
# context = f"Current time: {datetime.datetime.now()}"
# print(context)

# """
# The following is also equivalent:
# print(json.dumps({
#   "hookSpecificOutput": {
#     "hookEventName": "UserPromptSubmit",
#     "additionalContext": context,
#   },
# }))
# """

# # Allow the prompt to proceed with the additional context
# sys.exit(0)