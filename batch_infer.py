"""Backward-compatible batch inference entry point.

New code should prefer ``python model_tools.py infer ...``.  This wrapper keeps
the former ``--input-dir`` spelling and the former default model id.
"""

from __future__ import annotations

import sys

from efficientad_tools.cli import main as tools_main


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    translated = []
    for argument in arguments:
        if argument == "--input-dir":
            translated.append("--input")
        elif argument.startswith("--input-dir="):
            translated.append("--input=" + argument.split("=", 1)[1])
        else:
            translated.append(argument)

    if "--model" not in translated and not any(
        value.startswith("--model=") for value in translated
    ):
        translated.extend(["--model", "1"])
    return tools_main(["infer", *translated])


if __name__ == "__main__":
    raise SystemExit(main())
