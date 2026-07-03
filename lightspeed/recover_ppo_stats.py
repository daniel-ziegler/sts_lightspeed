#!/usr/bin/env python3

import re
import json
import argparse
import sys
from typing import Dict, Optional

def parse_ppo_log(log_file: str, output_file: Optional[str] = None):
    """
    Parse ppo_train log output and recover stats in JSONL format.
    
    Args:
        log_file: Path to the log file containing ppo_train output
        output_file: Optional path to write JSONL output (default: stdout)
    """
    
    # Regex patterns to extract stats from log lines
    # Note: Using [-+]?[\d.]+ to handle negative numbers
    patterns = {
        'iteration': re.compile(r'^Iteration (\d+)/\d+'),
        'trajectories_and_time': re.compile(r'^Collected (\d+) trajectories in ([\d.]+)s'),
        'win_rate_stats': re.compile(r'^Win rate: ([-+]?[\d.]+), Avg floor: ([-+]?[\d.]+), Avg reward: ([-+]?[\d.]+)'),
        'experiences': re.compile(r'^Training on (\d+) experiences'),
        'train_time': re.compile(r'^Training completed in ([\d.]+)s'),
        'losses': re.compile(r'^Policy loss: ([-+]?[\d.]+), Value loss: ([-+]?[\d.]+), Entropy: ([-+]?[\d.]+)'),
        'training_metrics': re.compile(r'^KL div: ([-+]?[\d.]+), Grad norm: ([-+]?[\d.]+), Clip frac: ([-+]?[\d.]+)')
    }
    
    current_stats = {}
    recovered_stats = []
    
    with open(log_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Try to match each pattern
            if match := patterns['iteration'].match(line):
                # New iteration - save previous stats if complete and start new
                if current_stats and 'iteration' in current_stats:
                    recovered_stats.append(current_stats.copy())
                current_stats = {'iteration': int(match.group(1))}
                
            elif match := patterns['trajectories_and_time'].match(line):
                current_stats.update({
                    'num_trajectories': int(match.group(1)),
                    'collect_time': float(match.group(2))
                })
                
            elif match := patterns['win_rate_stats'].match(line):
                current_stats.update({
                    'win_rate': float(match.group(1)),
                    'avg_floor': float(match.group(2)),
                    'avg_reward': float(match.group(3))
                })
                
            elif match := patterns['experiences'].match(line):
                current_stats['num_experiences'] = int(match.group(1))
                
            elif match := patterns['train_time'].match(line):
                current_stats['train_time'] = float(match.group(1))
                
            elif match := patterns['losses'].match(line):
                current_stats.update({
                    'policy_loss': float(match.group(1)),
                    'value_loss': float(match.group(2)),
                    'entropy': float(match.group(3))
                })
                
            elif match := patterns['training_metrics'].match(line):
                current_stats.update({
                    'kl_div': float(match.group(1)),
                    'grad_norm': float(match.group(2)),
                    'clipfrac': float(match.group(3))
                })
    
    # Don't forget the last iteration
    if current_stats and 'iteration' in current_stats:
        recovered_stats.append(current_stats)
    
    # Output results
    if output_file:
        with open(output_file, 'w') as f:
            for stats in recovered_stats:
                f.write(json.dumps(stats) + '\n')
        print(f"Recovered {len(recovered_stats)} iterations of stats to {output_file}")
    else:
        for stats in recovered_stats:
            print(json.dumps(stats))
    
    return recovered_stats

def validate_stats(stats_list):
    """Validate recovered stats and report any issues."""
    expected_fields = {
        'iteration', 'num_trajectories', 'collect_time', 'win_rate', 
        'avg_floor', 'avg_reward', 'num_experiences', 'train_time',
        'policy_loss', 'value_loss', 'entropy', 'kl_div', 
        'grad_norm', 'clipfrac'
    }
    
    issues = []
    for i, stats in enumerate(stats_list):
        missing_fields = expected_fields - set(stats.keys())
        if missing_fields:
            issues.append(f"Iteration {stats.get('iteration', '?')} (index {i}): missing fields {missing_fields}")
    
    if issues:
        print("Warning: Some stats may be incomplete:", file=sys.stderr)
        for issue in issues:
            print(f"  {issue}", file=sys.stderr)
    else:
        print(f"Successfully recovered complete stats for {len(stats_list)} iterations", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='Recover stats from ppo_train log output')
    parser.add_argument('log_file', help='Path to the log file containing ppo_train output')
    parser.add_argument('-o', '--output', help='Output JSONL file (default: stdout)')
    parser.add_argument('--validate', action='store_true', 
                        help='Validate recovered stats and report missing fields')
    
    args = parser.parse_args()
    
    try:
        stats_list = parse_ppo_log(args.log_file, args.output)
        
        if args.validate:
            validate_stats(stats_list)
            
    except FileNotFoundError:
        print(f"Error: Log file '{args.log_file}' not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing log file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()