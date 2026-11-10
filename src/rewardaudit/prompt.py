"""What a row's prompt asks for and what it offers, read from its text: the form of the answer (the answer only, first or
last, a letter, a quantity, an integer, a percentage), whether the order of a list matters, and the options, numbered
choices or names it lists."""

from __future__ import annotations

import re

from rewardaudit.types import Task

_SET_RE = re.compile(
    r"\b(?:in any order|any order|order (?:does not|doesn't) matter|regardless of (?:the )?order|unordered|"
    r"as a set|the set of (?:all )?(?:the )?(?:solutions|roots|zeros|values)|"
    r"all (?:of )?(?:the )?(?:(?:possible|real|integer|complex|distinct|positive|negative|rational) )*"
    r"(?:solutions|roots|zeros|zeroes)|"
    r"all (?:of )?(?:the )?(?:(?:possible|real|integer|complex|distinct|positive|negative|rational) )*values\b"
    r"(?! of \w+ and\b))", re.I)
_ORDERED_RE = re.compile(
    r"\b(?:ordered (?:pair|triple|tuple|list|sequence|set|quadruple|\w+-tuple)s?|"
    r"in (?!any\b|no\b)(?:\w+ ){0,2}order\b(?!\s+to\b)|sorted (?:list|order|in|by|from)\b|"
    r"sort (?:them|the (?:list|numbers|values|elements))\b|ascending|descending|"
    r"from (?:smallest|least|lowest|largest|greatest|highest) to|"
    r"(?:output|give|write|return|enter|list|print|provide|find) (?:a |the )?(?:\w+ )?sequence\b)", re.I)


def order_semantics(prompt: str) -> str | None:
    """'set' if the prompt says order does not matter, 'ordered' if it says order matters, else None
    (also when it says both)."""
    s, o = bool(_SET_RE.search(prompt)), bool(_ORDERED_RE.search(prompt))
    return "set" if s and not o else "ordered" if o and not s else None


_ANSWER_ONLY_RE = re.compile(
    r"\b(?:return|output|reply|respond|(?<!your final )answer|give|provide|write|state|print)\b"
    r"[^.\n]{0,30}?\b(?:only|just)\b|"
    r"\bonly\s+(?!one\b)(?:\w+\s+){0,3}?(?:answer|result|number|value|letter|code|option|choice)\b|"
    r"\b(?:answer|respond|reply)\s+(?:directly\b|with\b[^.\n]{0,60}?\bdirectly\b)|"
    r"\bnothing else\b|\bno (?:other|additional) text\b|\bno explanations?\b|\bwithout (?:any )?explanations?\b|"
    r"\bdo not (?:explain|include (?:any )?(?:explanation|other text|extra))|\bdon't explain", re.I)
# A sentence that limits the inside of a tag or box ('Do not explain your reasoning inside the answer tags; provide only
# the final answer') or asks for reasoning too ('reason step by step, then ONLY give the letter') leaves the reply free.
_REPLY_FREE_RE = re.compile(
    r"\b(?:do not|don't)\b.*\b(?:inside|within|in)\s+(?:the\s+|your\s+)?(?:\S+\s+){0,2}?(?:tags?\b|box\b|\\boxed)|"
    r"\bstep[- ]by[- ]step\b|\breasoning process\b|\bafter your explanation\b", re.I)


def _chat_text(task: Task) -> str:
    return "\n".join(str(m.get("content")) for m in task.chat())


def asks_answer_only(task: Task) -> bool:
    """The prompt asks for only the answer, so a right answer with prose around it may rightly be refused. Read
    sentence by sentence, since the pattern's \\s would otherwise join 'only' to an unrelated next line."""
    sentences = (s for line in _chat_text(task).splitlines() for s in re.split(r"(?<=[.!?])\s+", line))
    return any(_ANSWER_ONLY_RE.search(s) and not _REPLY_FREE_RE.search(s) for s in sentences)


_ANSWER_FIRST_RE = re.compile(r"\b(?:state|give|write|put|provide|output)\s+(?:the|your)\s+(?:final\s+)?"
                              r"(?:answer|amount|result|number|value|letter|option|choice)\s+first\b", re.I)
_ANSWER_LAST_RE = re.compile(
    r"\b(?:very\s+)?(?:last|final)\s+(?:line|thing)\s+(?:of\s+(?:your|the)\s+(?:response|reply|output|answer)\b"
    r"|you\s+(?:generate|write|output)\b)|"
    r"\bat\s+the\s+end\s+of\s+(?:your|the)\s+(?:response|reply|output|answer)\b|"
    r"\b(?:output|write|give|put|provide|state|place)\s+(?:the|your)\s+(?:final\s+)?answer\b[^.\n]{0,50}?\bat\s+the\s+end\b|"
    r"\b(?:response|reply|answer|output)\s+(?:should|must)\s+end\s+with\b", re.I)


def asks_answer_first(task: Task) -> bool:
    """The prompt asks for the answer at the start of the reply ('State the amount first'), so a right answer with text
    before it may rightly be refused; text after it is still allowed."""
    return bool(_ANSWER_FIRST_RE.search(_chat_text(task)))


def asks_answer_last(task: Task) -> bool:
    """The prompt asks for the answer at the end of the reply ('The last line of your response should be of the form
    ...', 'write a line ... at the end of your response'), so a right answer with text after it may rightly be refused;
    reasoning before it is what the prompt asks for."""
    return bool(_ANSWER_LAST_RE.search(_chat_text(task)))


def writes_two_blocks(task: Task) -> bool:
    """The prompt writes out the reply as a thinking block followed directly by an answer block ('<think> reasoning process
    here </think><answer> answer here </answer>')."""
    return bool(re.search(r"</think>\s*<[A-Za-z_][\w-]*>", _chat_text(task)))


# An instruction that names the letter as the answer; not 'letter' in a passage, nor '(x)' in a formula
_ASKS_LETTER_RE = re.compile(r"(?i:\b(?:answer|respond(?:ing)?|reply|choose|select|output|give|provide|write|put|tell)\b"
                             r"[^.\n]{0,80}?\bletter\b)|\$LETTER\b|\(X\)")


def asks_letter(prompt: str) -> bool:
    """The prompt asks for an option's letter ('Answer with the option's letter', 'Answer: $LETTER', 'The answer is (X)'):
    the letter is the answer there, and an option's text is not."""
    return bool(_ASKS_LETTER_RE.search(prompt))


_NUMBERED_LINE = re.compile(r"(?m)^[ \t]*\(?(\d{1,2})[ \t]*(?:-|\.|\))[ \t]+\S")


def numbered_choices(prompt: str) -> list[str]:
    """The numbers of the choices a prompt lists on numbered lines ('1 - piano', '2. desk'), counting up from 1, where it
    calls them choices or options; [] otherwise. A numbered list of steps or rules is not a list of choices."""
    if not re.search(r"\b(?:choices?|options?)\b", prompt, re.I):
        return []
    run: list[int] = []
    for n in map(int, _NUMBERED_LINE.findall(prompt)):
        run = [*run, n] if n == len(run) + 1 else [1] if n == 1 else run
    return [str(n) for n in run] if len(run) >= 2 else []


# A question that ends in a choice between names, after its first comma or colon ('Who is older, Adele or Taylor Swift?'),
# behind a label such as 'Question:'
_CHOICE_QUESTION = re.compile(r"(?:\w+:\s*)?[^,:]+[,:]\s*(.+?),?\s+or\s+([^,:?]+?)\s*\?\s*")


def _names(items: list[str]) -> list[str]:
    """The items, if each is a name: two letters in a row, at most eight words, no question mark, exclamation mark, colon or
    semicolon (a period may end an abbreviation: St. Lucia)."""
    items = [re.sub(r"^(?:or|and)\s+", "", x.strip()) for x in items]
    return items if all(re.search(r"[^\W\d_]{2}", x) and len(x.split()) <= 8 and not re.search(r"[?!;:]", x)
                        for x in items) else []


def listed_names(prompt: str) -> list[str]:
    """The names a prompt offers to choose from, from its last line that offers any: a line that is a list of three or more
    capitalised names ('East Timor, Samoa, Kenya'), or a question that ends in a choice ('Which is older, the Nile or the
    Amazon?'). A list of numbers or of single letters is not one of names; a line of prose is not a list."""
    for line in reversed(prompt.splitlines()):
        question = re.split(r"(?<=[.!?])\s+", line.strip())[-1]
        if m := _CHOICE_QUESTION.fullmatch(question):
            if len(names := _names([*m.group(1).split(","), m.group(2)])) >= 2:
                return names
        elif not line.strip().endswith("?"):
            names = _names(line.strip().rstrip(".").rsplit(":", 1)[-1].split(","))
            if len(names) >= 3 and all(x[0].isupper() for x in names):
                return names
    return []


_QUANTITY_RE = re.compile(
    r"\bhow (?:many|much|long|far|old|tall|high|fast|often)\b|\b(?:the|total) number of\b(?!\s+the\b)|"
    r"\banswer with (?:a|one|a single) number\b", re.I)
_IDENTIFIER_RE = re.compile(r"\b(?:versions?|index(?:es)?|indices|identifiers?|pass ?keys?|passcodes?)\b", re.I)


def asks_quantity(task: Task) -> bool | None:
    """True if the question asks for a quantity ('How many people', 'Answer with a number'), False if it names a version
    or an index, None if it says neither. Read from the prompt's last lines, which hold the question and the form of the
    answer; a passage before them can mention anything."""
    tail = "\n".join([ln for ln in task.prompt.splitlines() if ln.strip()][-3:])
    return True if _QUANTITY_RE.search(tail) else False if _IDENTIFIER_RE.search(tail) else None


_INTEGER_RE = re.compile(r"\b(?:answer|respond|reply|output|return|give|provide)\w*\b[^.\n]{0,30}\binteger\b"
                         r"|\b(?:as|must be|should be) an integer\b|\binteger only\b", re.I)


def asks_integer(task: Task) -> bool:
    """The form of the answer is an integer ('Answer with one integer'): 2.0 is then another form than the one asked for."""
    return bool(_INTEGER_RE.search("\n".join([ln for ln in task.prompt.splitlines() if ln.strip()][-3:])))


_ASKS_PERCENT = re.compile(r"\bwhat\s+(?:is\s+the\s+)?percent(?:age)?\b|\b(?:as|in)\s+(?:a\s+)?percent(?:age)?\b"
                           r"|\bhow\s+many\s+percent\b", re.I)


def question_part(prompt: str) -> str:
    """The end of a prompt, where the question usually is; a line on the format of the reply may follow it."""
    return " ".join(re.split(r"(?<=[.?!])\s+", prompt.strip())[-2:])


def asks_percent(prompt: str) -> bool:
    """The question wants a percentage, as opposed to mentioning one on the way ('he ate 20 percent of them')."""
    return bool(_ASKS_PERCENT.search(question_part(prompt)))


def mcq_options(prompt: str) -> list[str]:
    """Letters of options listed in the prompt, e.g. 'A. ...', '(B) ...', 'C) ...'."""
    letters = re.findall(r"(?m)^\s*\(?([A-J])[\.\):]\s+\S", prompt)
    seen: list[str] = []
    for ch in letters:
        if ch not in seen:
            seen.append(ch)
    return seen if len(seen) >= 2 else []


def option_texts(prompt: str) -> dict[str, str]:
    """Letter -> text of the options listed in the prompt. With few-shot examples above it, the last question's options win."""
    found = {letter: text for letter, text in re.findall(r"(?m)^[ \t]*\(?([A-J])[\.\):][ \t]+(\S.*?)[ \t]*$", prompt)}
    return found if len(found) >= 2 else {}
