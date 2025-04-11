# How to improve mathematical skills with AutoGen Agents

Large Language Models (LLMs) like GPT-4 have revolutionized the field of
natural language processing, enabling a wide range of applications from text
generation to complex problem-solving. However, despite their impressive
capabilities, LLMs still face significant challenges when it comes to
arithmetic and mathematical operations. This article explores some of the key
issues that LLMs encounter in these areas, provides examples of common
errors, and shows how an agentic approach can help provide better results.

The tutorial makes use of AutoGen and shows you how to improve mathematical
operations in two ways. The first is by giving a tool to an LLM allowing it
to perform simple calculations. The second is the implementation of an
agentic system that can perform various mathematical tasks. The code in this
tutorial can be found
[here](https://github.com/mzat-msft/tutorials/tree/main/llm-math).


## Problems

### Inherent limitations in next token prediction

One of the primary challenges that LLMs face in arithmetic operations is the
inherent limitation in next token prediction. LLMs are trained on vast
amounts of text data, which primarily consists of natural language rather
than precise numerical data. As a result, these models often struggle with
tasks that require exact numerical calculations, such as addition,
subtraction, multiplication, and division.

For example, when asked to perform a simple arithmetic operation like “What
is 1234 + 5678?”, an LLM might provide an incorrect answer such as “6911”
instead of the correct answer “6912”. This error occurs because the model
relies on pattern recognition rather than precise calculation.

### Contextual understanding of mathematical problems

LLMs excel at understanding and generating human language, but they often
struggle with the contextual understanding required for solving mathematical
problems. Mathematical problems frequently involve multiple steps and require
a deep understanding of the underlying concepts. LLMs, however, may not
always grasp the context or the logical sequence of steps needed to arrive at
the correct solution.

For example, consider a word problem: “If a train travels at 60 km/h for 2
hours and then at 80 km/h for 3 hours, what is the total distance traveled?”
An LLM might incorrectly calculate the total distance by simply adding the
speeds (60 + 80) and multiplying by the total time (5 hours), resulting in an
incorrect answer of 700 km instead of the correct answer of 340 km.

### Lack of specialized training data

Another significant challenge is the lack of specialized training data for
mathematical operations. While LLMs are trained on diverse datasets that
include a wide range of topics, they may not have sufficient exposure to
high-quality mathematical content. This lack of specialized training data can
hinder their ability to accurately perform mathematical operations and solve
complex problems.

> When asked to solve a quadratic equation like “Solve for $x$: $(x^2 - 5x +
> 6 = 0)$”, an LLM might provide an incorrect solution such as “"$x = 2$”
> without recognizing that there are two solutions: $(x = 2)$ and $(x = 3)$.

### Difficulty in handling symbolic mathematics

Symbolic mathematics, which involves manipulating mathematical symbols and
expressions, poses a unique challenge for LLMs. Unlike natural language,
symbolic mathematics requires precise and unambiguous manipulation of symbols
according to well-defined rules. LLMs, however, are not inherently designed
to handle symbolic manipulation, leading to errors and inconsistencies in
their outputs.

> When asked to simplify a mathematical expression like $((2x + 3)(x - 4))$,
> an LLM might incorrectly expand it to $2x^2 - 8x + 3x - 12$ instead
> of the correct expansion $2x^2 - 5x - 12$.

### Tokenization and its impact on arithmetic

Tokenization, the process of dividing input text into tokens, plays a crucial
role in how LLMs process and understand numerical data. Different
tokenization schemes can significantly impact the model's ability to perform
arithmetic operations accurately.

LLMs like GPT-3.5 and GPT-4 use different tokenization strategies for
numbers. Some models tokenize numbers digit by digit, while others use byte
pair encoding (BPE) to tokenize entire numbers or groups of digits. This
choice can lead to varying performance in arithmetic tasks. For instance,
right-to-left tokenization (enforced by comma separating numbers at inference
time) has been shown to improve performance in numerical reasoning tasks
[(read more about this topic here)](https://arxiv.org/abs/2502.08680).

## Potential solutions and future directions

Despite these challenges, there are several potential solutions and future
directions that can help improve the performance of LLMs in arithmetic and
mathematical operations:

### Reasoning techniques

One promising approach to addressing these challenges is the use of prompts
that steer the LLM to perform step-by-step reasoning. Chain-of- Thought (CoT)
reasoning [(more information about this topic
here)](https://arxiv.org/abs/2201.11903) involves generating a series of
intermediate reasoning steps that lead to the final answer. This method helps
LLMs break down complex problems into manageable parts, improving their
accuracy and reliability in arithmetic and mathematical tasks. Sampling
multiple chain-of-thought paths and then selecting the most consistent answer
is the approach explored by self-consistency [(more on this topic
here)](https://arxiv.org/abs/2203.11171).

Reasoning models released by OpenAI such as O1 or O3-mini have built-in
chain-of-thought capabilities, which do not require users to explicitly cue
for CoT in their prompting. See [this
article](https://techcommunity.microsoft.com/blog/Azure-AI-Services-blog/prompt-engineering-for-openai%E2%80%99s-o1-and-o3-mini-reasoning-models/4374010)
for a nice overview of how to get the best out of these models.

### Incorporating specialized training data

By incorporating more high-quality mathematical content into the training
datasets, researchers can help LLMs develop a better understanding of
mathematical concepts and improve their accuracy in performing arithmetic
operations. This can also be done by fine-tuning existing pre-trained models
[(more information on this topic here)](https://arxiv.org/abs/2402.00157).

### Developing hybrid models

Combining LLMs with specialized mathematical models or symbolic computation
engines can leverage the strengths of both approaches. Hybrid models can use
LLMs for natural language understanding and symbolic computation engines for
precise mathematical calculations.

### Enhancing contextual understanding

Improving the contextual understanding of LLMs through advanced training
techniques and fine-tuning can help them better interpret and solve complex
mathematical problems.

### Interactive problem solving

Implementing interactive problem-solving frameworks that allow users to guide
LLMs through the steps of a mathematical problem can enhance their accuracy
and reliability.

## Using agentic AI to perform calculations

Agentic AI refers to Artificial Intelligence systems that exhibit a degree of
autonomy and decision-making capabilities, often designed to perform tasks
without continuous human intervention. These systems can be particularly
beneficial in enhancing arithmetic computations within Large Language Models
(LLMs). By integrating agentic AI, LLMs can autonomously identify and execute
complex arithmetic operations, ensuring higher accuracy and efficiency. This
integration allows the models to handle a broader range of mathematical
problems, from basic calculations to advanced numerical analysis, thereby
improving their overall performance and reliability in tasks requiring
precise arithmetic computations.

To demonstrate, we implement an agent using
[AutoGen](https://microsoft.github.io/autogen/stable/index.html). We use
version 0.4.8.


### Prerequisites

To enable this tutorial, we first need to install the relevant packages:

```bash
pip install autogen-agentchat autogen-ext[openai,azure]
```

These packages are needed to use a model deployed on Azure OpenAI. We also
provide a
[requirements.txt](https://github.com/mzat-msft/tutorials/blob/main/llm-math/requirements.txt)
file with all the packages needed to run the scripts.

### Assistant agent

We start with a very simple example: Compute the sum of 30 random
numbers.

The implementation of this first agent is found in
[dumbagent.py](https://github.com/mzat-msft/tutorials/blob/main/llm-math/dumbagent.py).

AutoGen provides a built-in `AssistantAgent`, which is an agent that uses a
language model and can call tools. To provide the language model to the
agent, we first need to initialize it. As we are using an Azure OpenAI model,
we use the following code:

```python
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient


az_model_client = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o-mini",
    model="gpt-4o-mini",
    api_version="2024-06-01",
    azure_endpoint=os.getenv("AOAI_BASE"),  # Ensure you have an environment variable with the endpoint URL
    api_key=os.getenv("AOAI_KEY"),  # Ensure you have an environment variable with the model key
)
```

Now, we can initialize our first agent:

```python
from autogen_agentchat.agents import AssistantAgent


agent = AssistantAgent(
    name="assistant",
    model_client=az_model_client,
    system_message="You are a helpful assistant.",
)
```

Agents are invoked with the `on_messages()` method. There are different
message types in `AutoGen`, and we use `TextMessage` that implements a
simple text message. The method `on_messages()` also requires an argument
`cancellation_token`. This is a token used to cancel pending async calls.

```python
import random

from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken


nums = tuple(random.random() for _ in range(30))
response = await agent.on_messages(
    [TextMessage(content=f"Compute the sum of {nums}", source="user")],
    cancellation_token=CancellationToken(),
)
print(response.inner_messages)
print("LLM response: ", response.chat_message.content)
print("Correct response: ", sum(nums))
```

In this code block we first start by generating 30 random numbers.
We then ask the agent to sum them. You can see that `on_messages()` is an
async function, therefore it must be *awaited*. The agent response is then
printed to screen, and we also print the correct response computed with a
Python function.

If we run `dumbagent.py` we get the following output:

```bash
$ python dumbagent.py
[]
LLM response:  To compute the sum of the given numbers, we simply add them all together:

\[
0.98192629503311 + 0.44913959495134703 + 0.9550975758004904 + 0.7961467173217164 + 0.05701251483217984 + 0.821316523716076 + 0.20455770254746786 + 0.3544870158594513 + 0.2097364980871036 + 0.41825598576135026 + 0.4302912042336191 + 0.09734595524736045 + 0.9150327071487488 + 0.1753207520044029 + 0.13082725879931378 + 0.1409855854434413 + 0.6047648456845441 + 0.5697120143978236 + 0.37432774710364836 + 0.897823530659285 + 0.5037086325182719 + 0.3415878958302003 + 0.11112757177263288 + 0.5640722132312188 + 0.2537593920515281 + 0.621287705386636 + 0.5033159059617952 + 0.06904368251150672 + 0.9169453476732938 + 0.799897885675357
\]

Calculating this step-by-step or using a calculator will yield:

\[
\text{Sum} \approx 12.297741639823053
\]

Thus, the total sum of the given numbers is approximately **12.297741639823053**.
Correct response:  14.268854257244922
```

The first line in the output is an empty list `[]`. This is the result of
`print(response.inner_messages)`. What this tells us is that the agent is not
processing the task internally before responding. We will see later that it
will be different when we implement tools or other agents.

The agent goes into a lengthy monologue, but in the end, it gives us a wrong
response. This is expected as language models are not great with mathematical
tasks.

### Using chain-of-thought

As we mentioned earlier, can we use CoT to get the correct result? We can try
to change the prompt to be explicit in the step-by-step calculation. For
instance, we can ask the following:

```python
response = await agent.on_messages(
    [TextMessage(content=f"Compute the sum of {nums}. Do the computation in steps.", source="user")],
    cancellation_token=CancellationToken(),
)
```

We can find the modified prompt in
[cotagent.py](https://github.com/mzat-msft/tutorials/blob/main/llm-math/dumbagent.py).
If we run it, this is what we get:

```bash
$ python cotagent.py
[]
LLM response:  To compute the sum of the numbers provided, we'll break it down into steps for easier handling. Let's sum them in groups to ensure accuracy.

### Step 1: First group of numbers
1. **Adding the first 5 numbers:**
   - \( 0.14127228096717648 \)
   - \( 0.05659007188662635 \)
   - \( 0.41471264697545984 \)
   - \( 0.07227848538359027 \)
   - \( 0.6008866025458204 \)

   **Sum:**
   \[
   0.14127228096717648 + 0.05659007188662635 + 0.41471264697545984 + 0.07227848538359027 + 0.6008866025458204 = 1.2857409877586735
   \]

### Step 2: Second group of numbers
2. **Adding the next 5 numbers:**
   - \( 0.8379897543698347 \)
   - \( 0.8661225426358206 \)
   - \( 0.8178552345233704 \)
   - \( 0.3942907131888822 \)
   - \( 0.4904236984818413 \)

   **Sum:**
   \[
   0.8379897543698347 + 0.8661225426358206 + 0.8178552345233704 + 0.3942907131888822 + 0.4904236984818413 = 3.406682943199149
   \]

### Step 3: Third group of numbers
3. **Adding the next 5 numbers:**
   - \( 0.8612645437666976 \)
   - \( 0.048297412427538045 \)
   - \( 0.943380072339754 \)
   - \( 0.6018781446881311 \)
   - \( 0.9337550469930928 \)

   **Sum:**
   \[
   0.8612645437666976 + 0.048297412427538045 + 0.943380072339754 + 0.6018781446881311 + 0.9337550469930928 = 3.3885752202152137
   \]

### Step 4: Fourth group of numbers
4. **Adding the next 5 numbers:**
   - \( 0.9995048576992535 \)
   - \( 0.1657584837840922 \)
   - \( 0.9243308591422265 \)
   - \( 0.5209206541101584 \)
   - \( 0.09718646567109135 \)

   **Sum:**
   \[
   0.9995048576992535 + 0.1657584837840922 + 0.9243308591422265 + 0.5209206541101584 + 0.09718646567109135 = 2.907700320406822
   \]

### Step 5: Fifth group of numbers
5. **Adding the last 5 numbers:**
   - \( 0.5006860467589616 \)
   - \( 0.07545995919884119 \)
   - \( 0.6005575818167507 \)
   - \( 0.6312675413283965 \)
   - \( 0.04546446523135217 \)

   **Sum:**
   \[
   0.5006860467589616 + 0.07545995919884119 + 0.6005575818167507 + 0.6312675413283965 + 0.04546446523135217 = 1.853435594333302
   \]

### Final Sum
Now we will combine all the partial sums:
\[
1.2857409877586735 + 3.406682943199149 + 3.3885752202152137 + 2.907700320406822 + 1.853435594333302 = 12.841134066913158
\]

Thus, the total sum of the provided numbers is:
\[
\boxed{12.841134066913158}
\]
Correct response:  15.48758499076025
```

As we can see, CoT did not help here.

### Using a tool

The easiest way we can improve the assistant agent response is by providing a
tool to perform the calculation. If the agent can call an external
deterministic tool to perform the computation, then we should not have
problems with this type of task. The implementation of this section can be
found in
[sumagent.py](https://github.com/mzat-msft/tutorials/blob/main/llm-math/sumagent.py).

AutoGen allows agents to call Python functions as tools. We can define a
simple function that takes a Python list as an input and returns the sum of
its elements:

```python
async def sum_tool(nums: List[float]) -> float:
    """Return the sum of a list of numbers."""
    return sum(nums)
```

In AutoGen it is important to use type hinting correctly because the library
uses them to define the schema of the function call.

In order to let the agent know that it can use `sum_tool`, we can simply add
the function to the `tools` argument:

```python
agent = AssistantAgent(
    name="assistant",
    model_client=az_model_client,
    tools=[sum_tool],
    system_message="You are a helpful assistant.",
)
```

Now, when we run the `sumagent.py` script we get:

```bash
$ python sumagent.py
[ToolCallRequestEvent(source='assistant', models_usage=RequestUsage(prompt_tokens=361, completion_tokens=280), metadata={}, content=[FunctionCall(id='call_PfahtG6rBaTBDET6utH1u6RS', arguments='{"nums":[0.15607955739873258,0.8963597498678355,0.5747720719717742,0.07705616383780134,0.920007610464873,0.07992191409289973,0.15398267665658838,0.8461178057089017,0.00867008329992447,0.12367258544020898,0.7023358189598443,0.7187120138614209,0.22380835162473767,0.518359282649394,0.7033304226296623,0.5696660681794894,0.7402270995100922,0.6890434861371746,0.8316673167316022,0.16186907193369304,0.3756545128586557,0.43527780899456625,0.31130767988731833,0.8121905496837669,0.5752979063795476,0.01729136994895908,0.04571361421695874,0.9414824642739353,0.9323455485546613,0.488415163128753]}', name='sum_tool')], type='ToolCallRequestEvent'), ToolCallExecutionEvent(source='assistant', models_usage=None, metadata={}, content=[FunctionExecutionResult(content='14.630635768883772', name='sum_tool', call_id='call_PfahtG6rBaTBDET6utH1u6RS', is_error=False)], type='ToolCallExecutionEvent')]
LLM response:  14.630635768883772
Correct response:  14.630635768883772
```

Yes! Now the answer is correct. As promised, now the inner messages are much
more interesting. We see that the agent performs a call to the function, and
it passes the correct arguments.

### Using a team of agents

Until now, the things our agent can do are quite limited: It can only perform
summations. This is useful, but we can do more. AutoGen allows us to create a
much more powerful implementation by properly using an agentic approach. We
now implement a team of agents: Our assistant will be joined by an agent that
is able to use Python to perform computations! This allows the assistant to
solve mathematical tasks with the help of its new teammate. Our
implementation can be found in
[computeteam.py](https://github.com/mzat-msft/tutorials/blob/main/llm-math/computeteam.py).

AutoGen already implemented an agent that has access to the Python shell.
This agent reads the context and searches for Python code blocks and executes
them.

**WARNING**: Executing arbitrary code is very dangerous! For this reason,
this agent must be able to execute code in a restricted environment. AutoGen
allows us to define where to execute the code and provides a Docker executor,
so this is what we use.

We can initialize the code executor agent like this:

```python
from autogen_agentchat.agents import CodeExecutorAgent
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor


code_executor = DockerCommandLineCodeExecutor(work_dir="coding")
code_executor_agent = CodeExecutorAgent("code_executor", code_executor=code_executor)
await code_executor.start()
```

We need to `start()` the agent so that the Docker container is initialized
and we can use it to run Python code.

Now, we need to define the team structure. We use a very simple
implementation: `RoundRobinGroupChat`. This basically invokes each agent in a
sequential manner. We add the condition for which the team stops when the
code executor answers with a text message.
This is quite a simplistic implementation but introduces most of the basic
concepts for agentic AI.

```python
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat


# Stop the task if the code_executor_agent responds with a text message.
termination_condition = TextMessageTermination("code_executor")

# Create a team with the agents and the termination condition.
team = RoundRobinGroupChat(
    [assistant, code_executor_agent],
    termination_condition=termination_condition,
)
```

This is not enough. We need to instruct our assistant that to answer a
question it can output some Python code. The code executor takes the
assistant output and runs any Python code blocks that it finds. The assistant
is instructed like this:

```python
assistant = AssistantAgent(
    name="assistant",
    model_client=az_model_client,
    tools=[],
    system_message="""You are a helpful assistant.
    In case of mathematical questions solve them by writing python code.
    Do not write the result, only the python code.
    Ensure the code prints the answer""",
)
```

If we run `computeteam.py`, this is what we get:

````bash
$ python computeagent.py
user:  Compute the sum of (0.5210953350884949, 0.23249114206929122, 0.17993491326259625, 0.8610840589495627, 0.020270394086646437, 0.4773509280360936, 0.5149967679257246, 0.946033692152485, 0.44512526250216766, 0.3995888094343413, 0.26044296024116675, 0.9082139741241312, 0.9104223796757873, 0.07442095420712869, 0.10579447016639254, 0.7294204803776102, 0.8758922846837599, 0.26595890324402705, 0.7276235467680734, 0.4043764258947661, 0.538410201118577, 0.47519150551888933, 0.7873227394876985, 0.494946576256316, 0.0652947923325683, 0.010004581249110078, 0.053183652768027834, 0.25713836094124565, 0.08413890277102154, 0.17509531809134193)
assistant:  ```python
numbers = (0.5210953350884949, 0.23249114206929122, 0.17993491326259625,
           0.8610840589495627, 0.020270394086646437, 0.4773509280360936,
           0.5149967679257246, 0.946033692152485, 0.44512526250216766,
           0.3995888094343413, 0.26044296024116675, 0.9082139741241312,
           0.9104223796757873, 0.07442095420712869, 0.10579447016639254,
           0.7294204803776102, 0.8758922846837599, 0.26595890324402705,
           0.7276235467680734, 0.4043764258947661, 0.538410201118577,
           0.47519150551888933, 0.7873227394876985, 0.494946576256316,
           0.0652947923325683, 0.010004581249110078, 0.053183652768027834,
           0.25713836094124565, 0.08413890277102154, 0.17509531809134193)

result = sum(numbers)
print(result)
```
code_executor:  12.801264313425042

Correct response:  12.801264313425042
````

Let's see what's happening here. The task of computing the sum is given to the
assistant. The assistant writes a function to compute the answer and passes
the turn to the next agent in the team, which is the code executor agent. The
code executor runs the function and returns what the function prints. We get
the correct answer in the end.

With this approach we have traded simplicity for more flexibility. Using the tool
technique allows us to get the correct result, but at the expense of having
to write a tool for every capability. In fact, we would need to add a
`subtract_tool`, `multiply_tool`, and so on.
With the code executor approach, the architecture is bit more complex, but
we do not need to modify the system to add more operations.


## Conclusion

In this article, we have explored the challenges that LLMs
face with arithmetic and mathematical operations. Despite
their impressive capabilities in natural language processing, LLMs often
struggle with numerical precision, contextual understanding, and symbolic
mathematics. We have discussed several potential solutions to these challenges,
including chain-of-thought reasoning, incorporating specialized training
data, developing hybrid models, enhancing contextual understanding, and
implementing interactive problem-solving frameworks.

We have also demonstrated how an agentic approach, using tools like AutoGen, can
significantly improve the performance of LLMs in mathematical tasks. By
integrating agentic AI, we can create systems that autonomously identify and
execute complex arithmetic operations, ensuring higher accuracy and
efficiency.

Through practical examples, we have shown how to implement an agent that can
perform various mathematical tasks, from simple summations to more complex
computations, by leveraging the strengths of both LLMs and specialized
mathematical models. This hybrid approach not only enhances the capabilities
of LLMs but also opens up new possibilities for their application in fields
requiring precise arithmetic computations.

As we continue to develop and refine these techniques, we can look forward to
even more powerful and reliable AI systems that excel in both natural
language processing and mathematical problem-solving.

[Marco Zatta is on LinkedIn](https://www.linkedin.com/in/mzatta/)
