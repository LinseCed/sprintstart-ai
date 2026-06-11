from agents.base import Agent
from agents.synthesis_agent import SynthesisAgent
from agents.tools.agent_tool import AgentTool
from agents.tools.base import ToolRegistry
from llm.base import LLMClient
from store.base import VectorStore

_DECISION_ROLE = (
    "You coordinate specialist sub-agents to answer a developer's question. "
    "Delegate to the single most relevant capability by passing it a clear, "
    "self-contained task. Reply READY immediately, without calling any tool, only "
    "for greetings or meta questions that need no knowledge lookup."
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
        )
