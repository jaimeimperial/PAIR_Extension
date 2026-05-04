#!/usr/bin/env python3
"""
Batch runner for PAIR reproduction experiments.
Runs PAIR across all behaviors from JBB-Behaviors (100) and AdvBench subset (50).

"""

import subprocess
import csv
import os
import sys
import argparse
import time
import json
from pathlib import Path


# ─────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────

def load_jbb_behaviors(data_dir: str = "data") -> list[dict]:
    """Load JBB-Behaviors harmful behaviors CSV.
    Tries local file first, then downloads from HuggingFace."""
    
    csv_path = os.path.join(data_dir, "jbb_harmful_behaviors.csv")
    
    if not os.path.exists(csv_path):
        # Try downloading from HuggingFace
        print(f"JBB-Behaviors CSV not found at {csv_path}. Downloading...")
        os.makedirs(data_dir, exist_ok=True)
        url = "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/main/data/harmful-behaviors.csv"
        try:
            import urllib.request
            urllib.request.urlretrieve(url, csv_path)
            print(f"Downloaded to {csv_path}")
        except Exception as e:
            print(f"Failed to download JBB-Behaviors: {e}")
            print("Please download manually from: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors")
            sys.exit(1)
    
    behaviors = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            behaviors.append({
                "index": "jbb_" + row.get("Index", row.get("index", "")),
                "goal": row.get("Goal", row.get("goal", "")),
                "target": row.get("Target", row.get("target", "")),
                "category": row.get("Category", row.get("category", "")),
                "source": "jbb",
            })
    print(f"Loaded {len(behaviors)} JBB-Behaviors")
    return behaviors


def load_advbench_behaviors(data_dir: str = "data") -> list[dict]:
    """Load AdvBench custom 50-behavior subset CSV."""
    
    csv_path = os.path.join(data_dir, "harmful_behaviors_custom.csv")
    
    if not os.path.exists(csv_path):
        print(f"AdvBench CSV not found at {csv_path}.")
        print("This file should be in the data/ directory of the PAIR repo.")
        sys.exit(1)
    
    behaviors = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            behaviors.append({
                "index": "advbench_" + str(i),
                "goal": row.get("goal", row.get("Goal", "")),
                "target": row.get("target", row.get("Target", "")),
                "category": row.get("category", row.get("Category", "advbench")),
                "source": "advbench",
            })
    print(f"Loaded {len(behaviors)} AdvBench behaviors")
    return behaviors


# ─────────────────────────────────────────────
# Experiment runner
# ─────────────────────────────────────────────

def run_single_behavior(
    behavior: dict,
    attack_model: str,
    target_model: str,
    judge_model: str = "llama-guard-4",
    n_streams: int = 30,
    n_iterations: int = 3,
    use_jailbreakbench: bool = True,
    verbosity: int = 1,
) -> dict:
    """Run PAIR on a single behavior. Returns result dict."""
    
    goal = behavior["goal"]
    target_str = behavior["target"]
    category = behavior["category"]
    index = behavior["index"]
    source = behavior["source"]
    
    cmd = [
        sys.executable, "main.py",
        "--attack-model", attack_model,
        "--target-model", target_model,
        "--judge-model", judge_model,
        "--n-streams", str(n_streams),
        "--n-iterations", str(n_iterations),
        "--goal", goal,
        "--target-str", target_str,
        "--category", category,
        "--index", str(index).replace("jbb_", "").replace("advbench_", ""),
    ]
    
    if not use_jailbreakbench:
        cmd.append("--not-jailbreakbench")
    
    if verbosity > 0:
        cmd.append("-" + "v" * verbosity)
    
    print(f"\n{'='*60}")
    print(f"[{source.upper()}] Behavior {index}: {goal[:80]}...")
    print(f"  Attacker: {attack_model} -> Target: {target_model}")
    print(f"  Judge: {judge_model} | Streams: {n_streams} | Iters: {n_iterations}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min timeout per behavior
        )
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[-500:]}" if result.stderr else "")
            return {"index": index, "goal": goal, "status": "error", 
                    "error": result.stderr[-500:], "elapsed": elapsed}
        
        # Check stdout for jailbreak result
        jailbroken = "Found a jailbreak" in result.stdout or "Found a jailbreak" in result.stderr
        print(f"  {'JAILBROKEN' if jailbroken else 'NOT JAILBROKEN'} ({elapsed:.1f}s)")
        
        return {"index": index, "goal": goal, "status": "success",
                "jailbroken": jailbroken, "elapsed": elapsed, "source": source}
        
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"  TIMEOUT after {elapsed:.1f}s")
        return {"index": index, "goal": goal, "status": "timeout", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  EXCEPTION: {e}")
        return {"index": index, "goal": goal, "status": "exception", 
                "error": str(e), "elapsed": elapsed}


def run_experiment(
    behaviors: list[dict],
    attack_model: str,
    target_model: str,
    judge_model: str = "llama-guard-4",
    n_streams: int = 30,
    n_iterations: int = 3,
    use_jailbreakbench: bool = True,
    verbosity: int = 1,
    results_dir: str = "results",
    start_from: int = 0,
):
    """Run PAIR across all behaviors and save results."""
    
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(
        results_dir, 
        f"results_{attack_model}_vs_{target_model}_{judge_model}.jsonl"
    )
    
    # Load existing results to support resume
    completed_indices = set()
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed_indices.add(str(r["index"]))
                except:
                    pass
        print(f"Resuming: {len(completed_indices)} behaviors already completed")
    
    total = len(behaviors)
    jailbroken_count = 0
    completed_count = len(completed_indices)
    
    for i, behavior in enumerate(behaviors):
        if i < start_from:
            continue
        if str(behavior["index"]) in completed_indices:
            print(f"Skipping behavior {behavior['index']} (already completed)")
            continue
        
        result = run_single_behavior(
            behavior=behavior,
            attack_model=attack_model,
            target_model=target_model,
            judge_model=judge_model,
            n_streams=n_streams,
            n_iterations=n_iterations,
            use_jailbreakbench=use_jailbreakbench,
            verbosity=verbosity,
        )
        
        # Append result
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")
        
        completed_count += 1
        if result.get("jailbroken"):
            jailbroken_count += 1
        
        print(f"\n  Progress: {completed_count}/{total} | "
              f"Jailbroken: {jailbroken_count}/{completed_count} "
              f"({jailbroken_count/completed_count*100:.1f}%)")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"EXPERIMENT COMPLETE: {attack_model} vs {target_model}")
    print(f"  Total: {completed_count}/{total}")
    print(f"  Jailbroken: {jailbroken_count}/{completed_count} "
          f"({jailbroken_count/max(completed_count,1)*100:.1f}%)")
    print(f"  Results saved to: {results_file}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Batch runner for PAIR experiments")
    
    parser.add_argument("--experiment", choices=["repro", "extension"], default="repro",
                        help="Experiment type: repro (reproduction) or extension (new models)")
    parser.add_argument("--attack-model", default="mixtral",
                        help="Attacker model name")
    parser.add_argument("--target-model", default="gpt-3.5-turbo-1106",
                        help="Target model name")
    parser.add_argument("--judge-model", default="llama-guard-4",
                        help="Judge model name")
    parser.add_argument("--dataset", choices=["jbb", "advbench", "both"], default="both",
                        help="Which dataset(s) to run")
    parser.add_argument("--n-streams", type=int, default=30,
                        help="Number of parallel streams")
    parser.add_argument("--n-iterations", type=int, default=3,
                        help="Max iterations per behavior")
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing behavior CSVs")
    parser.add_argument("--results-dir", default="results",
                        help="Directory to save results")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Skip behaviors before this index")
    parser.add_argument("--verbosity", type=int, default=1,
                        help="Verbosity level (0=quiet, 1=info, 2=debug)")
    parser.add_argument("--test", action="store_true",
                        help="Quick test: 1 behavior, 3 streams, 2 iters")
    parser.add_argument("--not-jailbreakbench", action="store_true",
                        help="Bypass JailbreakBench target wrapper")
    
    args = parser.parse_args()
    
    # Determine if jailbreakbench should be used for target
    # Llama-4 is not supported by JBB, so disable it
    jbb_supported_targets = {
        "vicuna-13b-v1.5", "llama-2-7b-chat-hf", 
        "gpt-3.5-turbo-1106", "gpt-4-0125-preview"
    }
    use_jbb = args.target_model in jbb_supported_targets and not args.not_jailbreakbench
    
    if not use_jbb:
        print(f"NOTE: Target '{args.target_model}' not in JBB supported models. "
              f"Using --not-jailbreakbench.")
    
    # Load behaviors
    behaviors = []
    if args.dataset in ("jbb", "both"):
        behaviors.extend(load_jbb_behaviors(args.data_dir))
    if args.dataset in ("advbench", "both"):
        behaviors.extend(load_advbench_behaviors(args.data_dir))
    
    if not behaviors:
        print("No behaviors loaded. Check your data directory.")
        sys.exit(1)
    
    # Quick test mode
    if args.test:
        print("\n=== TEST MODE: 1 behavior, 3 streams, 2 iterations ===\n")
        behaviors = behaviors[:1]
        args.n_streams = 3
        args.n_iterations = 2
        args.verbosity = 2
    
    print(f"\nExperiment: {args.experiment}")
    print(f"Attacker: {args.attack_model}")
    print(f"Target: {args.target_model}")
    print(f"Judge: {args.judge_model}")
    print(f"JailbreakBench target wrapper: {use_jbb}")
    print(f"Streams: {args.n_streams} | Iterations: {args.n_iterations}")
    print(f"Behaviors to run: {len(behaviors)}")
    print()
    
    run_experiment(
        behaviors=behaviors,
        attack_model=args.attack_model,
        target_model=args.target_model,
        judge_model=args.judge_model,
        n_streams=args.n_streams,
        n_iterations=args.n_iterations,
        use_jailbreakbench=use_jbb,
        verbosity=args.verbosity,
        results_dir=args.results_dir,
        start_from=args.start_from,
    )


if __name__ == "__main__":
    main()
