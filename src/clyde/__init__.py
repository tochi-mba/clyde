"""clyde: the Claude Code CLI, wearing an OpenAI-compatible face.

One HTTP surface (`POST /v1/chat/completions`) over one subprocess (`claude -p`), so a host
that already speaks the chat-completions dialect can use a Claude Code subscription as its
model without knowing that is what it is doing.
"""

VERSION = "0.1.0"

__all__ = ["VERSION"]
