#!/usr/bin/env python3
"""
Simple script to run the Strands agent with Anthropic Claude.
"""

import asyncio
from strands_agent.agent import build_agent

async def main():
    """Run the agent with test questions."""
    print("🚀 Building Strands agent with Anthropic Claude...")
    
    try:
        agent = build_agent()
        print("✅ Agent built successfully!")
        
        # Test questions
        questions = [
            "What is the capital of France?",
            "Who wrote the novel '1984'?",
            "What is 2 + 2?",
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n🤔 Question {i}: {question}")
            try:
                response = await agent.invoke_async(question)
                print(f"💬 Answer: {response}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n🎉 Test completed!")
        
    except Exception as e:
        print(f"❌ Error building agent: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 