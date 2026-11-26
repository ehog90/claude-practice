import asyncio
import os
import sys
from utils import add_user_messages, add_assistant_message, chat


async def main(effort, stream):
    system_prompt = "Összegezd tömören és rövidebben. Ne formázd markuppal, konzolból nyomjuk."
    print("🤖 Chat initialized! Type 'exit' to quit.\n")
    if stream:
        print("📡 Streaming mode enabled.\n")

    messages = []

    while True:
        user_input = input("👤 You: ").strip()

        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        if not user_input:
            continue

        add_user_messages(messages, user_input)
        result = await chat(messages, system_prompt, effort=effort, stream=stream)
        add_assistant_message(messages, result)

        if not stream:
            print(f"🤖 Assistant: {result}\n")
        else:
            print("\n")


if __name__ == "__main__":
    stream = "--stream" in sys.argv

    effort_input = input("⚡ Choose effort level (low/medium/high/xhigh/max, default medium): ").strip().lower()
    valid_efforts = ["low", "medium", "high", "xhigh", "max"]
    if effort_input not in valid_efforts:
        effort = "medium"
        print(f"Invalid effort, using default: {effort}\n")
    else:
        effort = effort_input

    asyncio.run(main(effort, stream))

