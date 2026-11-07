"""Command line: audit | demo | probes."""

from __future__ import annotations

import re
from typing import Any, NoReturn

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.text import Text

from rewardaudit import __version__
from rewardaudit.api import DEFAULT_N
from rewardaudit.api import audit as run
from rewardaudit.errors import AuditError, WorkerError, short_error
from rewardaudit.pipeline import AuditResult, exit_code
from rewardaudit.report import render_terminal, write_outputs

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Test the checker of an RL environment, without a model.",
                  pretty_exceptions_enable=False)
console = Console()
errors = Console(stderr=True)
USAGE_ERROR = 2
RATES = ("non_solving_accepted", "wrong_accepted", "hedging_accepted", "right_rejected", "wrong_partial")


def _sample_size(n: str) -> int | str:
    if n == "all":
        return n
    try:
        size = int(n)
    except ValueError:
        size = 0
    if size < 1:
        raise AuditError(f"--n must be a positive whole number or 'all', got {n!r}")
    return size


def _budget(s: str | None) -> float | None:
    """Seconds from 90s, 10m, 1h, or a plain number of seconds."""
    if s is None:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", s.strip())
    if not m or float(m[1]) <= 0:
        raise AuditError(f"--budget must be a length of time such as 90s, 10m or 1h, got {s!r}")
    return float(m[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[m[2]]


def _fail_on(s: str | None) -> dict[str, float] | None:
    if not s:
        return None
    out: dict[str, float] = {}
    for part in s.split(","):
        key, _, val = part.strip().partition(">")
        if key.strip() not in RATES:
            raise AuditError(f"--fail-on does not know {key.strip()!r}; the rates are {', '.join(RATES)}")
        try:
            out[key.strip()] = float(val)
        except ValueError:
            raise AuditError(f"--fail-on expects metric>threshold, such as right_rejected>0.1, got {part!r}") from None
    return out


def _fail(message: str) -> NoReturn:
    errors.print(Text(f"Error: {message}", style="red"), soft_wrap=True)
    raise typer.Exit(USAGE_ERROR)


def _execute(call: Any, quiet: bool) -> AuditResult:
    """Run an audit under a progress bar. A problem the user can fix ends here with one message and exit code 2."""
    with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(),
                  console=console, transient=True, disable=quiet) as prog:
        bar = prog.add_task("starting", total=None)

        def progress(stage: str, done: int, total: int) -> None:
            prog.update(bar, description=stage, completed=done, total=total or None)  # None: no total, a pulsing bar

        def log(s: str) -> None:
            prog.console.print(Text(s, style="dim"))  # Text: third-party output is not console markup

        try:
            return call(log=None if quiet else log, progress=progress)
        except AuditError as e:
            message = str(e)
        except WorkerError as e:
            message = f"Could not load the checker: {short_error(str(e))}"
    _fail(message)


@app.command()
def audit(
    target: str = typer.Argument(..., help="reward.py[:function], a verifiers environment installed here, or owner/name from the Hub"),
    dataset: str | None = typer.Option(None, "--dataset", "-d", help="for a scoring function: jsonl, json, csv or parquet file, or hf:org/name[:config[:split]]"),
    n: str = typer.Option(str(DEFAULT_N), "--n", "-n", help="rows to draw at random from the whole dataset, or 'all'"),
    seed: int = typer.Option(0, help="seed of the random draw"),
    prompt_col: str | None = typer.Option(None, help="column holding the prompt (found by name if not given)"),
    answer_col: str | None = typer.Option(None, help="column holding the reference answer (found by name if not given)"),
    reference_col: str | None = typer.Option(None, help="column the checker receives as the reference, if not the answer column, such as a list of accepted answers"),
    out: str | None = typer.Option(None, help="directory for report.md, report.json and repro.py"),
    workers: int | None = typer.Option(None, help="parallel worker processes (default: 4, or the CPUs if fewer)"),
    timeout: float = typer.Option(10.0, help="seconds allowed per checker call"),
    budget: str | None = typer.Option(None, help="time the whole audit may take, such as 90s, 10m or 1h; past it no further probe is sent (default: no limit)"),
    probes: str | None = typer.Option(None, help="probe families to run, such as N,E,O (default: all)"),
    fail_on: str | None = typer.Option(None, help='exit 1 only above these rates, such as "non_solving_accepted>0.05,right_rejected>0.1"'),
    reply_format: str | None = typer.Option(None, "--format", help="how the checker wants a reply written, if it is none of the common formats, such as '<sol>{answer}</sol>'"),
    answer_field: bool = typer.Option(False, "--answer-field", help="the checker takes the reply to be its answer field, as a script scoring a file of predictions does; text around the answer is then not counted against it"),
    verifiers: str | None = typer.Option(None, help="for a Hub environment: constraint on the verifiers version, such as '<0.1.13'"),
    full: bool = typer.Option(False, "--full", help="also print the finding behind each defect, one per probe"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="print only the result"),
) -> None:
    """Audit one checker. Exit code: 0 no high defect, 1 such a defect, 2 the audit could not be done or the rates could not be measured."""
    try:
        size, gates, budget_s = _sample_size(n), _fail_on(fail_on), _budget(budget)
    except AuditError as e:
        _fail(str(e))
    result = _execute(lambda **kw: run(target, dataset, n=size, seed=seed, prompt_col=prompt_col, answer_col=answer_col,
                                       workers=workers, timeout=timeout, budget_s=budget_s,
                                       probes=probes.split(",") if probes else None,
                                       reference_col=reference_col, reply_format=reply_format, answer_field=answer_field,
                                       verifiers=verifiers, **kw), quiet)
    path = write_outputs(result, out) if out else None
    render_terminal(result, console, full=full, out=path)
    raise typer.Exit(exit_code(result, gates))


@app.command()
def demo(
    n: int = typer.Option(20, help="tasks per checker"),
    out: str | None = typer.Option(None, help="directory; one sub-directory per checker"),
    full: bool = typer.Option(False, "--full", help="also print the finding behind each defect"),
) -> None:
    """Audit three small built-in checkers that have deliberate defects."""
    from rewardaudit.demo import DEMOS

    for name in DEMOS:
        console.rule(f"demo: {name}")
        result = _execute(lambda name=name, **kw: run(f"demo:{name}", n=n, workers=1, **kw), True)
        render_terminal(result, console, full=full, out=write_outputs(result, f"{out}/{name}") if out else None)


@app.command()
def probes(
    key: str | None = typer.Argument(None, help="a probe (Q6) or a family (Q): show what it sends and what a sound checker does"),
) -> None:
    """List every probe, or explain one."""
    from rewardaudit.report import render_explain, render_probe_list

    if key is None:
        render_probe_list(console)
    elif not render_explain(key, console):
        raise typer.Exit(USAGE_ERROR)


def _version(value: bool) -> None:
    if value:
        console.print(f"rewardaudit {__version__}")
        raise typer.Exit()


@app.callback()
def _main(_version_flag: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show the version.")) -> None:
    pass


if __name__ == "__main__":
    app()
