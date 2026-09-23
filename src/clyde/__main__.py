"""`clyde serve`. Thin on purpose: everything worth testing is in `create_app`."""

from __future__ import annotations

import argparse

import uvicorn

from clyde.core.config import load_settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="clyde", description="Claude Code as an OpenAI provider")
    parser.add_argument("command", choices=["serve"], nargs="?", default="serve")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    settings = load_settings()
    uvicorn.run(
        "clyde.api.app:create_app",
        factory=True,
        host=args.host or settings.host,
        port=args.port or settings.port,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
