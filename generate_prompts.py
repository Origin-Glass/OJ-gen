import json
import random

tiers = ["브론즈", "실버", "골드", "플래티넘", "다이아몬드", "루비", "마스터"]
total_problems = 50000
problems_per_tier = total_problems // len(tiers)

tags_list = [
    ["구현", "수학"], ["그래프", "BFS", "DFS"], ["DP"], ["그리디"],
    ["문자열", "자료구조"], ["정렬", "이분탐색"], ["기하학", "완전탐색"]
]

output_file = "data/prompts_50k.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for tier in tiers:
        for _ in range(problems_per_tier):
            data = {
                "difficulty": tier,
                "tier_value": random.randint(1, 5),
                "tags": random.choice(tags_list),
                "time_limit": "2 seconds",
                "memory_limit": "256 MB"
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

print(f"Generated {output_file} with {problems_per_tier} problems per tier.")
