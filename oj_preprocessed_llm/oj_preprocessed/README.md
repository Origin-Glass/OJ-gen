# BOJ LLM preprocessing output

- `boj_problems_indexed.jsonl`: parsed BOJ records.
- `boj_sft_messages.jsonl`: ChatML-style SFT records.
- `boj_problems_indexed.csv`: compact index.
- `problem_ids.txt`: IDs for solved.ac enrichment.
- `stats.json`: counts.

Difficulty mapping: solved.ac level 0 = unrated, 1 = Bronze V, ..., 30 = Ruby I. solved.ac help states problem levels are contributor-assigned and may change over time, so cache the fetch date when enriching.
