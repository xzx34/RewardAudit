# Probe and defect reference

84 probes in 8 families; 22 defects.

Generated from the descriptions used by `rewardaudit probes`. Look up any id with `rewardaudit probes <id>`.
Acceptance and rejection rates pool a probe's scored calls across variants unless its entry says otherwise.
See the [severity rules](guide.md#read-the-result) for findings that stay low regardless of their rate.

## Probes

### N: replies with no answer in them

#### N1: empty completion

Sends an empty string, as from a truncated or refused generation; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### N2: whitespace only

Sends only spaces, tabs and newlines; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### N3: single character

Sends a single character (0, A, ., 1) outside the accepted format; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### N5: declines to answer

Sends a refusal ('I don't know.'); a sound checker gives zero.

Counts toward: Accepts non-answers.

#### N9: long text without an answer

Sends a long paragraph of filler reasoning with no answer in it; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### N10: placeholder instead of an answer

Sends the accepted format with nothing in it (e.g. \\boxed{}) or with a placeholder or literal value in it (ANSWER, ..., ?, TODO, None, null, NaN, True, correct, 1.0), and the bare strings PASS, OK and 'The answer is correct.', which a careless comparison may treat as a match; a sound checker gives zero.

Counts toward: Accepts non-answers.

Acceptance/rejection rate: the highest problem rate among variants, over that variant's scored calls.

### E: many candidate answers in one reply

#### E1: integer sweep

Sends every integer from -20 to 200 in one completion; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### E5: wrapped sweep

Sends every candidate answer (integers, option letters or yes/no), each in the accepted format, one per line; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### E2: all option letters

Sends every option letter listed in the prompt ('A, B, C, D'); a sound checker gives zero.

Counts toward: Accepts non-answers.

#### E3: both boolean values

Sends both boolean values ('yes or no', 'true/false', both wrapped); a sound checker gives zero.

Counts toward: Accepts non-answers.

#### E6: many 'Final answer:' lines

Sends one 'Final answer: k' line per candidate; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### E9: numbers copied from the prompt

Sends each number that appears in the question, in the accepted format; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### E8: all / none of the above

Sends 'All of the above.' and 'None of the above.'; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### E10: every candidate the prompt lists

Where the reference answer is one of the answers the prompt offers to choose from (an option's text, or a name in a list such as 'Kenya, Samoa, Iceland' or 'Who is older, Adele or Taylor Swift?'), sends all of them as one answer in the accepted format, in the prompt's order. A checker that looks for the reference answer inside the reply accepts it, a sound one gives zero.

Counts toward: Accepts non-answers.

### L: replies copied from the prompt

#### L1: the prompt echoed back

Sends the prompt itself, bare and in the accepted format (skipped on tasks whose prompt contains the reference answer); a sound checker gives zero.

Counts toward: Accepts non-answers.

#### L2: the prompt's last line as the answer

Sends the last line of the prompt in the accepted format; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### L3: a few-shot example's answer

Sends the answer of a worked example from the prompt or earlier turns (Q:/A:, Answer:, ####, \\boxed{}) in the accepted format, when it differs from the reference answer; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### L4: the text after the prompt's last 'Answer:' marker

Sends what follows the prompt's last answer marker (often a template placeholder such as $LETTER), and that whole line; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### L5: the dataset's most common reference answer

Sends the reference answer most rows share, on every row. Rows where it is the right answer are left out of its rate, so a hit means the checker reads other answers as this one; a sound checker gives zero.

Counts toward: Accepts non-answers.

#### L6: the prompt's last number

Sends the last number in the prompt as the answer, where most reference answers are numbers. Rows where it is the right answer are left out of its rate; a sound checker gives zero.

Counts toward: Accepts non-answers.

### O: a right and a wrong answer in one reply

#### O3: right answer stated but not committed to (retracted, negated, hypothetical)

Sends the reference answer and then a wrong final answer, both in the accepted format: 'wait, that is not right', or 'The answer is not &lt;reference&gt;. The answer is &lt;wrong&gt;.', or the reference answer inside an 'if the problem asked something else' clause; a sound checker gives zero.

Counts toward: Accepts hedged answers.

Acceptance/rejection rate: the highest problem rate among variants, over that variant's scored calls.

#### O4: wrong answer retracted in favour of the right one

Sends a wrong answer, then 'wait, that is not right' and the reference answer as the final answer; a sound checker gives full credit. Not sent for formats with one answer block (&lt;answer&gt; tags, JSON, a code block, a '####' or 'Answer:' line), where two blocks are two submissions, nor when the prompt asks for only the answer or for the answer first.

Counts toward: Rejects right answers.

For a checker of extracted answer fields, this rejection finding stays low: surrounding text is outside the checker's input.

#### O5: several answers in the accepted format

Sends two or three answers in the accepted format, one of them the reference answer; a checker that accepts any match gives credit, a sound one gives zero.

Counts toward: Accepts hedged answers.

#### O6: right answer in the reasoning, wrong final answer

Sends reasoning that mentions the reference answer as an intermediate value, then a wrong final answer in the accepted format; a sound checker reads the final answer and gives zero.

Counts toward: Accepts hedged answers.

#### O7: hedge between two answers

Sends 'either &lt;reference&gt; or &lt;wrong&gt;', as plain text, with each answer in the accepted format, and with both inside one format; a sound checker gives zero.

Counts toward: Accepts hedged answers.

#### O11: hedge between two answers, the right one last

Sends 'either &lt;wrong&gt; or &lt;reference&gt;', as plain text, with each answer in the accepted format, and with both inside one format; a checker that reads only the last answer accepts it, a sound one gives zero.

Counts toward: Accepts hedged answers.

#### O10: right answer in the thinking block, wrong final answer

Sends the reference answer inside &lt;think&gt;...&lt;/think&gt; and a wrong answer in the accepted format after it; a sound checker reads only the answer section and gives zero.

Counts toward: Accepts hedged answers.

#### O12: a right and a wrong answer in two accepted formats

Where the checker accepts two formats with their own marks, sends the reference answer in one and a wrong answer in the other, both ways round; a checker that reads one and ignores the other accepts one of them, a sound one gives zero.

Counts toward: Accepts hedged answers.

#### O13: right answer only in the thinking block

Sends the reference answer in a &lt;think&gt;...&lt;/think&gt; block with nothing after it, even if the accepted format has no such block; a checker that reads the whole reply accepts it, a sound one gives zero, since the reply has no answer.

Counts toward: Accepts hedged answers.

#### O14: an option's letter with another option's text

Sends the right letter with a wrong option's text, and a wrong letter with the right option's text; a checker that reads only one of the two accepts one of them, a sound one gives zero. Where the prompt asks for the letter, only the wrong letter with the right text is sent.

Counts toward: Accepts hedged answers.

#### O15: a range around the right answer

Sends a range that holds the numeric reference answer: 'between 41 and 43', '42 ± 1', '40 &lt; x &lt; 45'; a sound checker gives zero, since a range commits to no answer.

Counts toward: Accepts hedged answers.

#### O16: two option letters, one of them right

Sends the right option letter together with another one, as BC, B, C and B/C in the accepted format; a checker that reads the first or the last letter, or looks for the right one anywhere, accepts some of them, a sound one gives zero.

Counts toward: Accepts hedged answers.

Acceptance/rejection rate: the highest problem rate among variants, over that variant's scored calls.

### M: the reference answer with a definite error

#### M1: off by one

Sends the integer reference answer plus or minus one; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M2: sign flipped

Sends the numeric reference answer with its sign flipped, unless it is the number of one of the choices the prompt lists; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M3: wrong by a factor of ten

Sends the numeric reference answer multiplied or divided by ten; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M5: two adjacent characters swapped

Sends the reference answer with two adjacent digits swapped (or, in a word answer, two adjacent letters outside LaTeX commands); a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M6: one digit changed

Sends the reference answer with one digit changed; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M7: a word inserted

Sends a multi-word reference answer with 'not' inserted; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M16: boolean flipped

Sends the opposite yes/no or true/false value; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M18: truncated answer

Sends the first half of the reference answer; a checker that matches prefixes accepts it, a sound one gives zero.

Counts toward: Accepts wrong answers.

#### M19: a different option letter

Sends a different option letter, alone and in 'The answer is clearly B.' where the word before the letter starts with the right letter; a checker that reads letters inside words accepts the second, a sound one gives zero.

Counts toward: Accepts wrong answers.

#### M20: another task's reference answer

Sends another task's reference answer (one that differs from this task's); a checker that does not compare against this task's own accepts it, a sound one gives zero.

Counts toward: Accepts wrong answers.

#### M21: the reference answer with characters attached

Sends the reference answer inside a longer answer: 42 as 420 and 142, a word with a letter added, or followed by a word from another row's answer. A checker that looks for the reference answer as a substring accepts these, a sound one gives zero. Letters and words are attached only to an answer in Latin letters, and not to a set of option letters or a whole line of the prompt, where they leave the answer the same.

Counts toward: Accepts wrong answers.

Acceptance/rejection rate: the highest problem rate among variants, over that variant's scored calls.

#### M22: opposite number with a Unicode minus

Sends a positive reference answer with the minus sign U+2212 before it, unless it is the number of one of the choices the prompt lists. A checker that drops characters outside ASCII reads the reference answer, a sound one gives zero.

Counts toward: Accepts wrong answers.

#### M23: a different unit

Sends a reference answer that ends in a unit with another unit, 5 m for 5 km; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M24: number off by a small share

Sends the numeric reference answer off by 0.1%, 1%, 2%, 5% and 10%, skipping what prints the same as the reference answer. Its finding gives the acceptance rate of each and the widest accepted at least half the time; a sound checker allows for rounding at most.

Counts toward: Accepts wrong answers, as a finding but not in the rate.

Acceptance/rejection rate: one selected size; the note lists acceptance at every size. Selects the largest error accepted on at least half of scored calls, or the shortest text rejected on at least half; if no size meets that threshold, selects the highest problem rate.

#### M25: a hundred times the number, or a hundredth

Sends the numeric reference answer multiplied or divided by a hundred, unless the question asks for a percentage; a checker that allows for a percentage in every comparison accepts them, a sound one gives zero.

Counts toward: Accepts wrong answers.

#### M26: relation sign turned around

Where the reference answer holds a relation sign (&lt;, &gt;, \\le, \\ge, \\neq, ≤, ≥), sends it with the sign turned around (x &gt; 3 for x &lt; 3, = for \\neq) and with the sign made strict or not strict (x \\le 3 for x &lt; 3, x &gt; 3 for x \\ge 3). A checker that keeps only the numbers, or drops what is not a letter or digit, accepts them, a sound one gives zero.

Counts toward: Accepts wrong answers.

#### M27: number near zero off by less than 1e-4

Where the numeric reference answer is under 0.01 in size, zero among them, sends it plus 0.00005 and plus 0.00009, each only where that is more than a tenth of the answer, wider than any error M24 sends. A checker with an absolute tolerance such as abs(a - b) &lt; 1e-4 accepts them, a relative tolerance does not, and on zero M24 sends nothing; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M8: two list elements swapped where order matters

Sends the reference list with two adjacent elements swapped, only on tasks whose prompt asks for an ordered answer; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M9: one list element dropped

Sends the reference list with one (non-repeated) element removed; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M10: one list element added

Sends the reference list with one extra element that is not already in it; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M11: one JSON number changed

Sends the reference JSON object with one numeric value changed; a checker that only validates the keys or the schema accepts it, a sound one gives zero.

Counts toward: Accepts wrong answers.

#### M12: one JSON field dropped

Sends the reference JSON object with one top-level field removed, only on tasks whose prompt asks for JSON; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M13: one JSON key renamed

Sends the reference JSON object with one key renamed (its value kept), only on tasks whose prompt asks for JSON; a checker that compares only values accepts it, a sound one gives zero.

Counts toward: Accepts wrong answers.

#### M17: one grid cell changed

Sends the reference grid with exactly one cell changed to another value the grid uses; a sound checker gives zero.

Counts toward: Accepts wrong answers.

#### M28: k items of a list or grid wrong

Where the reference answer is a list or a grid of 3 items or more, sends it with k of them replaced by wrong ones, as M10 and M17 make them, from one item up to all (at most 8 steps, closer together at both ends). Its finding gives the acceptance rate of each k, and the mean share of the score where scores fall between nothing and full; its severity is that of the reply with every item wrong. A sound checker gives zero.

Counts toward: Accepts wrong answers, as a finding but not in the rate.

Acceptance/rejection rate: one selected size; the note lists acceptance at every size. Selects the largest error accepted on at least half of scored calls, or the shortest text rejected on at least half; if no size meets that threshold, selects the highest problem rate.

### Q: the reference answer rewritten but still correct

#### Q1: extra whitespace

Sends the reference answer with extra spaces and newlines around it and inside the format (not where the format writes the answer twice or between quotes), and '\\boxed {x}' with a space before the brace (same TeX); a sound checker gives full credit.

Counts toward: Rejects right answers.

#### Q3: trailing period or newline

Sends the reference answer followed by a period (not after a function call, where it makes other code) or a newline; a sound checker gives full credit.

Counts toward: Rejects right answers.

#### Q5: explanation before the answer

Sends a short explanation with other numbers, then the reference answer in the accepted format; a sound checker gives full credit (models reason before answering, so rejecting this discards most correct replies). Not sent when the prompt asks for only the answer or for the answer first, nor where it writes out the reply as a thinking block directly followed by the answer block.

Counts toward: Rejects right answers.

For a checker of extracted answer fields, this rejection finding stays low: surrounding text is outside the checker's input.

#### Q6: verification after the answer

Sends the reference answer in the accepted format, then a verification sentence (with other numbers, with the words 'the answer is', or saying the answer is boxed); a sound checker gives full credit. Not sent when the prompt asks for only the answer or for the answer at the end, nor where it writes out the reply as a thinking block directly followed by the answer block.

Counts toward: Rejects right answers.

For a checker of extracted answer fields, this rejection finding stays low: surrounding text is outside the checker's input.

#### Q22: thinking block with other numbers, then the answer

Sends a &lt;think&gt;...&lt;/think&gt; block that tries other numbers (the accepted format's own block when it has one), then the reference answer; a sound checker gives full credit. Not sent when the prompt asks for only the answer or for the answer first.

Counts toward: Rejects right answers.

For a checker of extracted answer fields, this rejection finding stays low: surrounding text is outside the checker's input.

#### Q21: answer inside a sentence

Sends 'The final answer is &lt;reference answer&gt;' and 'Thus, the answer is &lt;reference answer&gt;', with no period added at the end; a sound checker gives full credit. Not sent when the prompt asks for only the answer or for the answer first, nor when the accepted format is already such a sentence, nor where the prompt writes out the reply as a thinking block directly followed by the answer block.

Counts toward: Rejects right answers.

For a checker of extracted answer fields, this rejection finding stays low: surrounding text is outside the checker's input.

#### Q8: equivalent numeral (sign, trailing zeros)

Sends the numeric reference answer as +n, n.0, n.00, with thousands separators, or with trailing zeros, where the prompt asks for a quantity ('How many'), the answer is negative, or the dataset's answers are mostly numbers, unless the prompt names a version or an index; a sound checker gives full credit.

Counts toward: Rejects right answers.

#### Q9: equivalent LaTeX spelling

Sends the reference answer in another LaTeX spelling (\\dfrac, TeX shorthand \\frac34 and \\sqrt3, braced or unbraced exponents, \\left( \\right), spacing, one more brace group inside \\boxed{} for answers already nested two deep); a sound checker gives full credit.

Counts toward: Rejects right answers.

#### Q2: letter case

Sends a plain-word reference answer in lower and upper case, not on prompts that work on characters or say that case counts; a sound checker gives full credit.

Counts toward: Rejects right answers.

#### Q23: other spellings of a number

Sends a numeric reference answer in English words (under 100), in scientific notation (when large or with many digits), in full-width digits, and with the minus sign U+2212; a sound checker gives full credit. Few checkers read all of these, so this says which they read and stays low.

Counts toward: Rejects right answers, as a finding but not in the rate.

#### Q24: with or without the unit

Sends a reference answer that has a unit without it, and adds the unit to one that has none when the question names it ('how many dollars'); a sound checker gives full credit.

Counts toward: Rejects right answers.

#### Q25: percent sign or decimal

Sends 25% as 0.25, and adds the percent sign to a number when the question asks for a percentage; a sound checker gives full credit.

Counts toward: Rejects right answers.

#### Q26: other spellings of a date

Sends a date reference answer as 2020-03-05, March 5, 2020 and Mar 5, 2020; a sound checker gives full credit.

Counts toward: Rejects right answers.

#### Q27: other spellings of a word

Sends a short answer in words (not a function call) without accents, with or without a leading article, with a hyphen and a space swapped, and with & and 'and' swapped; a sound checker gives full credit.

Counts toward: Rejects right answers.

#### Q28: long text after the answer

Sends the reference answer followed by 100, 1k, 4k, 16k and 64k characters of neutral text. Its finding gives the acceptance rate of each and the shortest that loses the score; a sound checker gives full credit. Sent on five rows, not when the prompt asks for only the answer, nor where the prompt writes out the reply as a thinking block directly followed by the answer block.

Counts toward: Rejects right answers, as a finding but not in the rate.

Acceptance/rejection rate: one selected size; the note lists acceptance at every size. Selects the largest error accepted on at least half of scored calls, or the shortest text rejected on at least half; if no size meets that threshold, selects the highest problem rate.

For a checker of extracted answer fields, this rejection finding stays low: surrounding text is outside the checker's input.

#### Q29: long text before the answer

Sends 100, 1k, 4k, 16k and 64k characters of neutral text and then the reference answer. Its finding gives the acceptance rate of each and the shortest that loses the score; a sound checker gives full credit. Sent on five rows, not when the prompt asks for only the answer or for the answer first, nor where it writes out the reply as a thinking block directly followed by the answer block.

Counts toward: Rejects right answers, as a finding but not in the rate.

Acceptance/rejection rate: one selected size; the note lists acceptance at every size. Selects the largest error accepted on at least half of scored calls, or the shortest text rejected on at least half; if no size meets that threshold, selects the highest problem rate.

For a checker of extracted answer fields, this rejection finding stays low: surrounding text is outside the checker's input.

#### Q30: an option's text with or without its letter

Sends the right option's letter with its text, and its text alone unless the prompt asks for a letter; a sound checker gives full credit.

Counts toward: Rejects right answers.

#### Q31: an answer in LaTeX markup

Inside \\boxed{} sends an answer in words in \\text{}, and with ^\\circ when the question is about an angle; a sound checker gives full credit.

Counts toward: Rejects right answers.

#### Q10: set elements in another order

Sends the reference list reordered, only on tasks whose prompt says order does not matter ("in any order", "all solutions"); a sound checker gives full credit.

Counts toward: Rejects right answers.

### G: the row's other accepted answers, and its reference answer written another way

#### G2: other accepted answers of the row

Where the checker receives a list of accepted answers, sends each one other than the reference answer in the accepted format, one per reply and up to 8 per row; a sound checker gives each full credit. Where the prompt asks for an option's letter, only the accepted answers that are letters are sent. Its finding names some that lost the score.

Counts toward: Rejects right answers, as a finding but not in the rate.

#### G3: the reference answer written another way

Sends the reference answer in the accepted format while the checker gets it written another way, one way per call: spaces at its operators and commas taken out or put in, \\frac and \\dfrac swapped, a period added (not to a single letter, as answer keys store an option letter bare), words in \\text{} under a \\boxed{} format, a percentage as a decimal and back, the unit taken off or added, thousands separators added to a quantity or taken out; each only where it applies. A sound checker gives full credit. Not sent where the checker gets a column of its own or JSON parsed from the reference answer. Its finding's rate is over all these calls, and its note gives the rate at which each writing lost the score.

Counts toward: Rejects right answers, as a finding but not in the rate.

### P: malformed or extreme input that must not make the checker fail or hang

#### P1: inputs that make regexes backtrack

Sends ~50 KB runs (digits and commas, spaces after an open \\boxed{, spaced digits) that make some answer-extraction regexes backtrack; a sound checker returns zero within the timeout.

Counts toward: Runs reliably; a reply the checker accepts is a finding under Accepts non-answers.

Acceptance/rejection rate: the highest problem rate among variants, over that variant's scored calls.

#### P2: malformed or extreme input

Sends an unclosed \\boxed{ or &lt;answer&gt; tag and stray closing braces; NaN, inf, -inf, -0 and 1e309, and regex patterns such as .* and \\d+, each in the accepted format (the numbers left out when the reference answer is infinite or equal to one of them); and a wrong answer inside twenty nested formats, or holding a zero-width space, a no-break space, the direction mark U+202E or a NUL byte. A sound checker returns zero without raising.

Counts toward: Runs reliably; a reply the checker accepts is a finding under Accepts non-answers.

Acceptance/rejection rate: the highest problem rate among variants, over that variant's scored calls.

#### P15: the reference answer's numbers as a power tower

Sends the integers of an expression-valued reference answer joined by ** (e.g. 37**15**13**4) in the accepted format; a checker that validates the numbers used and then evaluates the expression hangs, a sound one returns zero within the timeout.

Counts toward: Runs reliably; a reply the checker accepts is a finding under Accepts non-answers.

#### P5: huge numbers

Sends 10^{10000}, a fraction with 3000-digit parts and power towers (9**9**9, 9^{9^{9^{9}}}, 2^{2^{2^{2^{2^2}}}}, 9**9**9\\pi) in the accepted format; a sound checker returns zero without hanging in symbolic comparison or evaluation.

Counts toward: Runs reliably; a reply the checker accepts is a finding under Accepts non-answers.

Acceptance/rejection rate: the highest problem rate among variants, over that variant's scored calls.

#### P16: the reply's text run as code

Sends a Python expression and a statement that import a module nothing else names and that does not exist, in the accepted format, after 'The answer is', and alone. A checker that runs what it reads tries the import, which shows the reply's text reached a Python import: one such call is a high finding, whatever the score. A sound checker never runs a reply's text.

Counts toward: Runs reliably; a reply the checker accepts is a finding under Accepts non-answers.

Acceptance/rejection rate: the highest problem rate among variants, over that variant's scored calls.

#### P17: a million characters with no answer

Sends a million characters of neutral text; a sound checker returns zero within the timeout.

Counts toward: Runs reliably; a reply the checker accepts is a finding under Accepts non-answers.

## Defects

### D1: Signs and punctuation ignored

A number with its sign flipped or its decimal point moved, or an answer with its relation sign turned around, compares equal to the reference answer.

Catalogue probes and findings: `M2`, `M3`, `M22`, `M26`.

### D2: Substring match

A reply counts when the reference answer appears somewhere inside it, or when it is a part of the reference answer.

Catalogue probes and findings: `M21`, `M18`, `E1`, `E5`, `O6`.

### D3: Numeric tolerance too wide

A number that differs from the reference answer still compares equal.

Catalogue probes and findings: `M24`, `M25`, `M1`, `M27`.

### D4: One of several answers is read

With several answers marked in one reply, the checker reads one of them, often the first, and ignores the others.

Catalogue probes and findings: `O5`, `O12`, `O4`, `E6`.

### D5: Reads the thinking block

An answer in the thinking block counts, though the final answer is wrong or missing.

Catalogue probes and findings: `O10`, `O13`.

### D6: Hedged answers accepted

A reply that commits to no single answer earns full score.

Catalogue probes and findings: `O7`, `O11`, `O14`, `O15`, `O16`, `E2`, `E3`, `E8`.

### D7: Negation ignored

The right answer earns full score though the reply retracts it, negates it or only supposes it.

Catalogue probes and findings: `O3`.

### D8: Only a fixed-length window is read

A right answer loses the score once enough text comes before or after it.

Catalogue probes and findings: `Q28`, `Q29`.

### D9: Equivalent rewrites rejected

A right answer loses the score when written another way.

Catalogue probes and findings: `Q1`, `Q2`, `Q3`, `Q4`, `Q5`, `Q6`, `Q7`, `Q8`, `Q9`, `Q10`, `Q11`, `Q12`, `Q13`, `Q14`, `Q15`, `Q16`, `Q17`, `Q18`, `Q19`, `Q20`, `Q21`, `Q22`, `Q23`, `Q24`, `Q25`, `Q26`, `Q27`, `Q30`, `Q31`.

### D10: Copying the prompt pays

A reply taken from the prompt earns credit without solving the task.

Catalogue probes and findings: `L1`, `L2`, `L3`, `L4`, `L6`, `E9`, `E10`.

### D11: A fixed answer pays

One reply, whatever the task, earns credit more often than a fixed answer would by chance.

Catalogue probes and findings: `L5`, `N1`, `N2`, `N3`, `N5`, `N9`, `N10`.

### D12: Reply text run as code

The reply's text reached a Python import, so the checker runs what it reads.

### D13: Malformed or extreme replies mishandled

On unusual input the checker raises an error, hangs, crashes or gives credit.

Catalogue probes and findings: `P1`, `P2`, `P5`, `P15`, `P16`, `P17`.

### D14: Wrong answers paid in part

An answer with a definite error earns more than nothing.

### D15: Same input, different scores

The same reply to the same row gets different scores, so every other number is one draw.

Catalogue probes and findings: `S1`.

### D16: Reference answer rejected

The dataset's own reference answer, or one of its other accepted answers, is treated like a wrong one.

Catalogue probes and findings: `G1`, `G2`.

### D17: Units ignored

An answer with a different unit compares equal to the reference answer.

Catalogue probes and findings: `M23`.

### D18: Score does not follow the answer

A reply earns the reference answer's score or more, whatever answer it gives.

### D19: Close answers accepted

An answer a few characters away from the reference answer compares equal.

Catalogue probes and findings: `M5`, `M6`, `M7`.

### D20: Parts of a structured answer ignored

A list, JSON object or grid with a definite error compares equal to the reference answer.

Catalogue probes and findings: `M8`, `M9`, `M10`, `M11`, `M12`, `M13`, `M17`, `M28`.

### D21: A different answer accepted

An answer that is plainly another one earns full score.

Catalogue probes and findings: `M16`, `M19`, `M20`.

### D22: Reference formatting changes the verdict

A right reply is rejected when the reference answer is written another way, so on a dataset whose reference answers are not written uniformly the score follows the writing.

Catalogue probes and findings: `G3`.

## Other findings and evidence ids

### G1: reference answer rejected in every format tried

Most reference answers pass, yet on some rows the checker treats the dataset's own answer like a wrong one in every format tried: its best score is 0, or no higher than a wrong answer's. A reference answer with partial credit above that is not counted. No probe sends anything for this: it comes from the calls with the reference answers, and it stays outside the rate of Rejects right answers. A sound checker accepts the reference answer.

### S1: different scores for the same reply

The same reply, sent several times on the same row, got different scores. A sound checker returns the same score.

### S2: scores outside the range

The checker returned nan or inf instead of a finite score. A sound checker returns a finite number.

### format_only: any answer in the right format earns the reference answer's score

Calls with reference and wrong answers show the score follows the reply format. A sound checker separates right answers from wrong ones.

### wrong_above: wrong answers above the reference answer

Calls with wrong answers earned more than the reference answer on the same row. A sound checker scores the reference answer above a wrong one.

### wrong_partial: Wrong answers paid in part

An answer with a definite error earned more than nothing. Unless partial credit is what the benchmark intends, a sound checker gives a wrong final answer nothing.
