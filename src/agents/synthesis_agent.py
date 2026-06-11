from agents.base import Agent
from agents.tools.base import ToolRegistry
from agents.tools.fetch_file import FetchFileTool
from agents.tools.grep import GrepTool
from agents.tools.retrieve import RetrieveTool
from llm.base import LLMClient
from store.base import VectorStore

_DECISION_ROLE = (
    "You decide whether you have enough context to answer a developer's question "
    "about their team's knowledge base. Prefer retrieve for conceptual questions, "
    "grep for exact identifiers, and fetch_file when a filename is known."
)

_ANSWER_SYSTEM = """\
You are a helpful assistant answering questions about a developer's knowledge base.
Answer based solely on the provided sources. If they do not contain enough
information to answer fully, say so explicitly rather than guessing.
Be concise and precise. Use markdown formatting where appropriate.
"""


class SynthesisAgent(Agent):
    def __init__(self, llm: LLMClient, store: VectorStore) -> None:
        tools = ToolRegistry(
            [
                RetrieveTool(llm, store),
                GrepTool(store),
                FetchFileTool(store),
            ]
        )
        super().__init__(
            name="synthesis",
            description=(
                "Answer questions using the indexed knowledge base "
                "(docs, code, retros, tickets)."
            ),
            llm=llm,
            tools=tools,
            decision_role=_DECISION_ROLE,
            answer_system=_ANSWER_SYSTEM,
        )
