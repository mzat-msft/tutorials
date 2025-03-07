import asyncio
import os
import random
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from dotenv import load_dotenv

load_dotenv()

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o-mini",
    model="gpt-4o-mini",
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AOAI_BASE"),
    api_key=os.getenv("AOAI_KEY"),
)


async def sum_tool(nums: List[float]) -> float:
    """Return the sum of a list of numbers."""
    return sum(nums)


agent = AssistantAgent(
    name="assistant",
    model_client=az_model_client,
    tools=[sum_tool],
    system_message="You are a helpful assistant.",
)


async def assistant_run() -> None:
    nums = tuple(random.random() for _ in range(30))
    response = await agent.on_messages(
        [TextMessage(content=f"Compute the sum of {nums}", source="user")],
        cancellation_token=CancellationToken(),
    )
    print(response.inner_messages)
    print("LLM response: ", response.chat_message.content)
    print("Correct response: ", sum(nums))


asyncio.run(assistant_run())
