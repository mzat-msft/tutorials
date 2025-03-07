import asyncio
import os
import random

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_agentchat.conditions import TextMessageTermination
from dotenv import load_dotenv


load_dotenv()

az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o-mini",
    model="gpt-4o-mini",
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AOAI_BASE"),
    api_key=os.getenv("AOAI_KEY"),
)


assistant = AssistantAgent(
    name="assistant",
    model_client=az_model_client,
    tools=[],
    system_message="""You are a helpful assistant.
    In case of mathematical questions solve them by writing python code.
    Do not write the result, only the python code.
    Ensure the code prints the answer""",
)


async def assistant_run() -> None:
    nums = tuple(random.random() for _ in range(30))
    code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
    code_executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)
    await code_executor.start()

    # Stop the task if the code_executor_agent responds with a text message.
    termination_condition = TextMessageTermination("code_executor")

    # Create a team with the agents and the termination condition.
    team = RoundRobinGroupChat(
        [assistant, code_executor_agent],
        termination_condition=termination_condition,
    )

    response = await team.run(task=f"Compute the sum of {nums}")
    for message in response.messages:
        print(f"{message.source}: ", message.content)
    print("Correct response: ", sum(nums))
    await code_executor.stop()


asyncio.run(assistant_run())
