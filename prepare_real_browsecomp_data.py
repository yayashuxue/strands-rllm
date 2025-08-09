#!/usr/bin/env python3
"""
Create realistic BrowseComp-style test data for evaluation.
These are example browsing tasks that simulate real BrowseComp problems.
"""

import json
import argparse
from pathlib import Path


def create_realistic_browsecomp_data():
    """Create realistic BrowseComp-style test data."""
    
    tasks = [
        {
            "id": 0,
            "question": "Find the current price of the latest iPhone model on Apple's official website",
            "answer": "iPhone 15 Pro starts at $999",
            "problem_topic": "Technology",
            "difficulty": "Easy"
        },
        {
            "id": 1,
            "question": "What is the current population of Tokyo, Japan according to official government statistics?",
            "answer": "Approximately 14 million people",
            "problem_topic": "Geography",
            "difficulty": "Medium"
        },
        {
            "id": 2,
            "question": "Find the release date of the movie 'Inception' directed by Christopher Nolan",
            "answer": "July 16, 2010",
            "problem_topic": "Entertainment",
            "difficulty": "Easy"
        },
        {
            "id": 3,
            "question": "What is the current exchange rate between US Dollar and Euro?",
            "answer": "1 USD = approximately 0.92 EUR",
            "problem_topic": "Finance",
            "difficulty": "Medium"
        },
        {
            "id": 4,
            "question": "Find the winner of the 2023 Nobel Prize in Physics",
            "answer": "Pierre Agostini, Ferenc Krausz, and Anne L'Huillier",
            "problem_topic": "Science",
            "difficulty": "Medium"
        },
        {
            "id": 5,
            "question": "What is the current weather in New York City?",
            "answer": "Current temperature and conditions in NYC",
            "problem_topic": "Weather",
            "difficulty": "Easy"
        },
        {
            "id": 6,
            "question": "Find the latest version of Python programming language",
            "answer": "Python 3.12.x",
            "problem_topic": "Technology",
            "difficulty": "Easy"
        },
        {
            "id": 7,
            "question": "What is the current stock price of Tesla (TSLA)?",
            "answer": "Current TSLA stock price",
            "problem_topic": "Finance",
            "difficulty": "Medium"
        },
        {
            "id": 8,
            "question": "Find the recipe for traditional Italian pizza margherita",
            "answer": "Pizza margherita recipe with ingredients and instructions",
            "problem_topic": "Food",
            "difficulty": "Medium"
        },
        {
            "id": 9,
            "question": "What is the current world record for the 100-meter sprint?",
            "answer": "9.58 seconds by Usain Bolt",
            "problem_topic": "Sports",
            "difficulty": "Easy"
        }
    ]
    
    return tasks


def save_to_jsonl(tasks, output_path):
    """Save tasks to JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for task in tasks:
            f.write(json.dumps(task) + '\n')
    
    print(f"Saved {len(tasks)} realistic BrowseComp tasks to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create realistic BrowseComp test data")
    parser.add_argument("--output", type=str, default="data/realistic_browsecomp.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of tasks to create")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate realistic tasks
    all_tasks = create_realistic_browsecomp_data()
    
    # Apply limit if specified
    if args.limit:
        all_tasks = all_tasks[:args.limit]
    
    # Save to JSONL
    save_to_jsonl(all_tasks, args.output)
    
    # Print sample
    print("\nSample tasks:")
    for task in all_tasks[:3]:
        print(f"- ID {task['id']}: {task['question'][:60]}...")


if __name__ == "__main__":
    main() 