from agents.base import Agent
from agents.synthesis_agent import SynthesisAgent
from agents.tools.agent_tool import AgentTool
from agents.tools.base import ToolRegistry
from llm.base import LLMClient
from store.base import VectorStore

_MAX_STEPS = 3

_DECISION_ROLE = (
    "You coordinate specialist sub-agents to answer a developer's question. "
    "For any question that seeks information, facts, or knowledge — including "
    "vague or underspecified ones — you MUST delegate to the single most relevant "
    "sub-agent by passing it a clear, self-contained task. Never answer such a "
    "question from your own knowledge and never ask the developer to clarify; "
    "delegate with the question as given instead. Delegate to each agent at most "
    "once for a question — never re-delegate to an agent you have already used; "
    "call a different agent only when it is clearly needed. Reply directly without "
    "calling any tool only for bare greetings, thanks, or small talk."
)

_ANSWER_SYSTEM = """\
You are SprintStart's assistant for software teams.
Answer the developer's question using the information gathered by your sub-agents.
If nothing relevant was gathered, answer from general knowledge but say so.
Be concise and precise. Use markdown formatting where appropriate.
"""


class OrchestratorAgent(Agent):
    def __init__(self, llm: LLMClient, store: VectorStore) -> None:
        tools = ToolRegistry(
            [
                AgentTool(SynthesisAgent(llm, store)),
            ]
        )
        super().__init__(
            name="orchestrator",
            description="Routes a question to the right specialist sub-agent.",
            llm=llm,
            tools=tools,
            decision_role=_DECISION_ROLE,
            answer_system=_ANSWER_SYSTEM,
            max_steps=_MAX_STEPS,
        )
