"""Click-based CLI entry point. Subcommands fill in across later milestones."""

from __future__ import annotations

import click

from chatqec import __version__
from chatqec.api import ChatQEC


@click.group(context_settings={"help_option_names": ["-h", "--help"]})  # type: ignore[misc]
@click.version_option(__version__, prog_name="chatqec")  # type: ignore[misc]
def main() -> None:
    """ChatQEC — open-source QEC research assistant."""


@main.command()  # type: ignore[misc]
@click.argument("question", nargs=-1, required=True)  # type: ignore[misc]
def ask(question: tuple[str, ...]) -> None:
    """Answer a single QEC question (one-shot)."""
    bot = ChatQEC()
    answer = bot.ask(" ".join(question))
    click.echo(answer.text)


if __name__ == "__main__":
    main()
