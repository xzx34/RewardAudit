# Confirmed defects in named checkers

In 2026-11, RewardAudit 0.3.0 was run on 185 checkers copied at pinned commits.
Each item below was checked by reading the upstream source at that commit and reproducing the recorded score.
These are selected clear cases, not every defect found; sections are ordered by their number of items.
Each item describes the code at the linked commit, which may since have changed.
Please open an issue if an item is wrong or has been fixed. Some outputs are shortened with “…”; “\n” denotes a newline.

## D1: Signs and punctuation ignored

- [DataBench](https://github.com/jorses/databench_eval) — [Evaluator.default_compare](https://github.com/jorses/databench_eval/blob/65f92cfad4f986f5496170f6bbcee48323f56c2f/src/databench_eval/eval.py#L52-L58) (numeric answers).
  Filtering to digits, dots and ASCII minus signs deletes the Unicode minus, making a negative path length equal its positive reference.
  Output: `−234.7`; reference answer: `234.7`; score: `1.0`.

- [AllenNLP Models](https://github.com/allenai/allennlp-models) — [_tokenize](https://github.com/allenai/allennlp-models/blob/b1f372248c17ad12684d344955fbcd98e957e77e/allennlp_models/rc/tools/drop.py#L40-L53) (DROP).
  Splitting on hyphens removes the minus sign before numeric normalization, so a negative percentage difference receives full F1.
  Output: `-43.2`; reference answer: `43.2`; score: `1.0`.

- [HybridQA](https://github.com/wenhuchen/HybridQA) — [normalize_answer](https://github.com/wenhuchen/HybridQA/blob/db22fda8c5951438fade3c69d75b350335ba93b3/evaluate_script.py#L7-L24).
  Punctuation removal deletes the minus sign, making a negative campus count equal the positive reference.
  Output: `-5`; reference answer: `5`; score: `2.0` (EM `1.0` + F1 `1.0`).

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — [exact_match_hf_evaluate](https://github.com/EleutherAI/lm-evaluation-harness/blob/d6de81643928d653435c431bae19945d41d32520/lm_eval/api/metrics.py#L254-L266) (MGSM German and Chinese; the metric is also used by GSM8K, Natural Questions and TriviaQA).
  With punctuation removal enabled for MGSM, the metric deletes the extracted minus sign and accepts a negative number of hours.
  Output: `Die Antwort lautet -3.`; reference answer: `3`; score: `1.0`.

- [LongBench](https://github.com/THUDM/LongBench) — [normalize_answer](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/LongBench/metrics.py#L12-L28) (HotpotQA; also used by NarrativeQA, Qasper, MultiFieldQA-en, 2WikiMultihopQA, MuSiQue and TriviaQA).
  Removing the decimal point makes a different numeric value equal the required year.
  Output: `200.6`; reference answer: `2006`; score: `100.0` (percentage scale).

- [MuSiQue](https://github.com/StonyBrookNLP/musique) — [normalize_answer](https://github.com/StonyBrookNLP/musique/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/metrics/answer.py#L12-L29).
  Removing the decimal point turns a wrong numeric value into the reference year.
  Output: `196.4`; reference answer: `1964`; score: `1.0`.

- [OpenCompass](https://github.com/open-compass/opencompass) — [gsm8k_postprocess](https://github.com/open-compass/opencompass/blob/6288ea0db61150b6c2c1e37c01808fef66748e8c/opencompass/datasets/gsm8k.py#L44-L49) (GSM8K).
  The numeric pattern skips the Unicode minus and extracts the positive magnitude from a negative student count.
  Output: `\boxed{−30}`; reference answer: `30`; score: `100.0` (percentage scale).

- [Open Instruct](https://github.com/allenai/open-instruct) — [GSM8KVerifier.__call__](https://github.com/allenai/open-instruct/blob/8e7a3314d832c8cea844bf97613c901802d1d91c/open_instruct/ground_truth_utils.py#L228-L236) (GSM8K).
  The numeric pattern recognizes ASCII signs only, so a Unicode negative payment is read as the positive reference.
  Output: `Answer: −11`; reference answer: `11`; score: `1.0`.

- [Platinum Benchmarks](https://github.com/MadryLab/platinum-benchmarks) — [get_parse_fn.parse_fn_math](https://github.com/MadryLab/platinum-benchmarks/blob/8fd2f82e63c49ea1cca4266f4dded82b7ddbcb55/src/utils.py#L45-L48) (GSM8K).
  The numeric search skips the Unicode minus and scores a negative distance difference as the positive reference.
  Output: `Final Answer: −80`; reference answer: `80`; score: `1.0`.

- [MMMU](https://github.com/MMMU-Benchmark/MMMU) — [extract_numbers](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py#L100-L120) (open questions).
  The numeric patterns skip the Unicode minus, making a negative rectifier rating match the positive reference.
  Output: `The final answer is −50.`; reference answer: `50`; score: `1.0`.

- [simple-evals](https://github.com/openai/simple-evals) — [parse_answer](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/mgsm_eval.py#L83-L94) (MGSM).
  Unsigned numeric extraction removes the minus sign, accepting a negative number of hours for a positive reference.
  Output: `Antwort: -3`; reference answer: `3`; score: `1.0`.

- [SkyThought](https://github.com/NovaSky-AI/SkyThought) — [extract_answer](https://github.com/NovaSky-AI/SkyThought/blob/0d190f11fd8e885bbe113aeccacba5ccde5b1102/skythought/evals/util/math_parsing_util.py#L281-L288) (GSM8K; the extractor is also used by its MATH500, MinervaMATH, AIME and AMC23 handlers).
  The last-number pattern skips the Unicode minus, turning a negative student count into the positive reference.
  Output: `Final Answer: −30`; reference answer: `30`; score: `1.0`.

- [TableBench](https://github.com/TableBench/TableBench) — [normalize_answer](https://github.com/TableBench/TableBench/blob/6c61a20340409dea8f96edb8675e66bb4abd7b28/metrics/qa_metrics.py#L25-L41).
  Punctuation removal deletes the minus sign before numeric comparison, accepting a negative total number of films.
  Output: `Final Answer: -1062`; reference answer: `1062`; score: `100.0` (percentage scale).

## D6: Hedged answers accepted

- [AGIEval](https://github.com/ruixiangcui/AGIEval) — [find_first_capital_letter](https://github.com/ruixiangcui/AGIEval/blob/84ab72d94318290aad2e4ec820d535a95a1f7552/src/post_process.py#L72-L78) (LogiQA-zh).
  The fallback returns the first option letter and ignores the unresolved alternative.
  Output: `The answer is either D or A.`; reference answer: `D`; score: `1.0`.

- [GAOKAO-Bench](https://github.com/OpenLMLab/GAOKAO-Bench) — [extract_choice_answer](https://github.com/OpenLMLab/GAOKAO-Bench/blob/6dbb24f8d8439041e5431c4c184a582182a6ce9c/Bench/bench_function.py#L49-L53) (Math I single choice).
  Taking the last A–D letter accepts an unresolved choice between two answers.
  Output: `The answer is either A or D.`; reference answer: `D`; score: `1.0`.

- [LegalBench](https://github.com/HazyResearch/legalbench) — [evaluate_sara_numeric_acc](https://github.com/HazyResearch/legalbench/blob/b46bf4ffae90524b2b72aaa30e7745fe9db64481/evaluation.py#L278-L288) (SARA numeric).
  The first number in an either/or answer is scored while the incompatible alternative is ignored.
  Output: `The answer is either $174 or $7611.`; reference answer: `$174`; score: `1.0`.

- [simple-evals](https://github.com/openai/simple-evals) — [GPQAEval.__call__.fn](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/gpqa_eval.py#L59-L61) (GPQA).
  The first matching answer prefix supplies the score even when the output offers two alternative choices.
  Output: `The answer is either Answer: B or Answer: A.`; reference answer: `B`; score: `1.0`.

- [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) — [match_str](https://github.com/UKGovernmentBEIS/inspect_ai/blob/e253d18ea2c0a3ea1c7030c354aa442ab7268e8d/src/inspect_ai/scorer/_common.py#L89-L115) (GSM8K, numeric matching).
  Reading the last numeric word accepts a hedge between two incompatible totals.
  Output: `The answer is either 60 or 30.`; reference answer: `30`; score: `1.0`.

## D7: Negation ignored

- [CMMLU](https://github.com/haonan-li/CMMLU) — [extract_choice](https://github.com/haonan-li/CMMLU/blob/d6e7b716d8ac694f38969a6c0407437d1fded799/src/mp_utils.py#L123-L143).
  The first matching answer phrase is returned even when the output explicitly retracts it and chooses another option.
  Output: `答案是：C … Wait, that is not right. … The correct result is 答案是：A`; reference answer: `C`; score: `1.0`.

- [grade-school-math](https://github.com/openai/grade-school-math) — [extract_answer](https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L24-L35) (GSM8K).
  The first marked number is retained after the output explicitly retracts it and gives a wrong replacement.
  Output: `#### 30 … Wait, that is not right. … The correct result is #### 60`; reference answer: `30`; score: `1.0`.

- [LightEval](https://github.com/huggingface/lighteval) — [gsm8k_normalizer](https://github.com/huggingface/lighteval/blob/c4826ea9380f01fd7d6cfe291cee26a238dad774/src/lighteval/metrics/normalizations.py#L341-L359) (GSM8K).
  The first marked number survives normalization despite an explicit retraction and a different final answer.
  Output: `#### 30 … Wait, that is not right. … The correct result is #### 60`; reference answer: `30`; score: `1.0`.

- [KOR-Bench](https://github.com/KOR-Bench/KOR-Bench) — [extract_text_from_brackets](https://github.com/KOR-Bench/KOR-Bench/blob/bb8194dff8e51e2e2247cf62aa9932c3d57d53d7/eval/eval_utils.py#L63-L78) (operation; also used by cipher).
  The first bracketed answer is selected even after the output explicitly retracts it for a different final answer.
  Output: `[[-435-308i]] … Wait, that is not right. … The correct result is [[x=2,y=3]]`; reference answer: `[[-435-308i]]`; score: `1.0`.

- [LongBench](https://github.com/THUDM/LongBench) — [extract_answer](https://github.com/THUDM/LongBench/blob/2e00731f8d0bff23dc4325161044d0ed8af94c1e/pred.py#L58-L68) (LongBench v2).
  The first matching answer phrase is returned despite its explicit retraction and a different final choice.
  Output: `The correct answer is (C) … Wait, that is not right. … The correct result is The correct answer is (A)`; reference answer: `C`; score: `1.0`.

## D2: Substring match

- [Inspect AI](https://github.com/UKGovernmentBEIS/inspect_ai) — [includes](https://github.com/UKGovernmentBEIS/inspect_ai/blob/e253d18ea2c0a3ea1c7030c354aa442ab7268e8d/src/inspect_ai/scorer/_match.py#L54-L60) (InfiniteBench passkey task).
  Substring containment accepts a different six-digit key containing the five-digit reference.
  Output: `\boxed{478690}`; reference answer: `47869`; score: `1.0`.

- [simple-evals](https://github.com/openai/simple-evals) — [fuzzy_match](https://github.com/openai/simple-evals/blob/652c89d0ca9df547706735883097e9537d40dc47/drop_eval.py#L215-L222) (DROP fuzzy score).
  Bidirectional substring matching accepts a different number containing the reference digit.
  Output: `Answer: 14`; reference answer: `4`; score: `1.0`.

- [MuSR](https://github.com/Zayne-sprague/MuSR) — [main](https://github.com/Zayne-sprague/MuSR/blob/b1f4d4168a9cfc6760e8b74d728e4516023dfaa5/eval/eval.py#L214-L225) (object placements).
  Testing whether the reference choice digit occurs in the answer accepts choice 20 although the listed choices are 1 through 4.
  Output: `ANSWER: 20`; reference answer: `choice 2: desk`; score: `1.0`.

## D12: Reply text run as code

- [Enigmata](https://github.com/BytedTsinghua-SIA/Enigmata) — [verify](https://github.com/BytedTsinghua-SIA/Enigmata/blob/b9c6f59f6d3f2ca97dd019bc02ca8f622575cf86/verifiable_tasks/tasks/game24/verifier.py#L38-L59) (24 game).
  The fenced output is passed to unrestricted Python evaluation before the allowed numbers are checked.
  Output: fenced code block containing `__import__('rewardaudit_probe_marker')`; reference answer: `((2*8)-4)+12 (inputs 2, 8, 4, 12)`; score: `0.0`; the Python import was attempted.

- [Reasoning360](https://github.com/LLM360/Reasoning360) — [_check_single_answer](https://github.com/LLM360/Reasoning360/blob/13158341d2a0dfe5f3bb80e7126ff21de0d16676/verl/utils/reward_score/tablereason.py#L6-L13) (table reasoning / FinQA).
  Extracted answer text reaches unrestricted Python evaluation before conversion to a number.
  Output: `\boxed{__import__('rewardaudit_probe_marker')}`; reference answer: `0.2146`; score: `0.0`; the Python import was attempted.

- [TheoremQA](https://github.com/wenhuchen/TheoremQA) — [number_it](https://github.com/wenhuchen/TheoremQA/blob/585c960f647b3cae8b37ff71fd9dac45ce4b4e91/few_shot_eval/number_utils.py#L51-L73).
  When number parsing fails, the fallback evaluates the output text as unrestricted Python.
  Output: `The answer is __import__('rewardaudit_probe_marker')`; reference answer: `1850.0`; score: `0.0`; the Python import was attempted.

## D3: Numeric tolerance too wide

- [DART-Math](https://github.com/hkust-nlp/dart-math) — [EvaluatorMath.is_num_eq](https://github.com/hkust-nlp/dart-math/blob/219b32467b55f7c610195e148b56df3431e1a963/dart_math/eval.py#L978-L999) (MATH; also used for DeepMind Mathematics).
  The percentage-equivalence branch accepts one hundredth of an ordinary count of clock chimes.
  Output: `\boxed{0.78}`; reference answer: `78`; score: `1.0`.

- [OlympiadBench](https://github.com/OpenBMB/OlympiadBench) — [MathJudger.numerical_equal](https://github.com/OpenBMB/OlympiadBench/blob/ba5b26a7e2849940b598a9159c1190daa2b9175f/inference/code/math_judger.py#L170-L189).
  Hundredfold percentage alternatives are applied to an exact-integer question whose answer must be at most 1,000.
  Output: `\boxed{97900}`; reference answer: `979`; score: `1.0`.

## D4: One of several answers is read

- [LiveBench](https://github.com/LiveBench/LiveBench) — [mathcontest_process_results](https://github.com/LiveBench/LiveBench/blob/1d1006a0d10ece52c2e706f8a208a659b5bbb96f/livebench/process_results/math/math_competitions/utils.py#L18-L20) (math competitions).
  Finding the reference letter repeated four times anywhere gives full credit to several conflicting answer strings.
  Output: `AAAAA\nBBBBB\nAAAAA`; reference answer: `B`; score: `1.0`.

## D9: Equivalent rewrites rejected

- [InfiniteBench](https://github.com/OpenBMB/InfiniteBench) — [first_int_match](https://github.com/OpenBMB/InfiniteBench/blob/51d9b37b0f1790ead936df2243abbf7f0420e439/src/compute_scores.py#L122-L129) (passkey retrieval).
  The first digit run in an explanatory preamble replaces the correct final passkey.
  Output: `First, identify the quantities involved. … (for instance 21 and 48 appear as intermediate values) … Therefore, \boxed{47869}`; reference answer: `47869`; score: `0.0`.

## D10: Copying the prompt pays

- [OpenAI Evals](https://github.com/openai/evals) — [fuzzy_match](https://github.com/openai/evals/blob/8eac7a7de5215c907fbddc30efdaf316913eccdd/evals/elsuite/utils.py#L46-L53) (largest_country).
  Substring matching gives full accuracy to a list copied from all ten candidates without selecting the largest country.
  Output: `\boxed{Kyrgyzstan, Philippines, Panama, Suriname, New Caledonia, Grenada, Netherlands Antilles, Western Sahara, Somalia, China}`; reference answer: `China`; score: `1.0`.

## D19: Close answers accepted

- [LiveBench](https://github.com/LiveBench/LiveBench) — [zebra_puzzle_process_results](https://github.com/LiveBench/LiveBench/blob/1d1006a0d10ece52c2e706f8a208a659b5bbb96f/livebench/process_results/reasoning/zebra_puzzle/utils.py#L103-L122) (zebra puzzles).
  Substring matching counts an explicitly negated nationality as correct, giving all four fields full credit.
  Output: `<solution>not canadian, 2, chef, thai</solution>`; reference answer: `canadian, 2, chef, thai`; score: `1.0`.
