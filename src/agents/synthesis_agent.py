from collections.abc import Generator

from agents.base import Agent, AgentRunState
from agents.tools.base import Invocation, ToolRegistry
from agents.tools.fetch_file import FetchFileTool
from agents.tools.grep import GrepTool
from agents.tools.retrieve import RetrieveTool
from llm.base import LLMClient
from rag.query_expansion import expand_query
from rag.retriever import retrieve
from store.base import VectorStore

_MAX_STEPS = 3

_SEED_TOP_K = 5
_SEED_MIN_SCORE = 0.3
_SEED_EXTRA_QUERIES = 2

_DECISION_ROLE = (
    "You decide whether you have enough context to answer a developer's question "
    "about their team's knowledge base. Prefer retrieve for conceptual questions "
    "and grep for exact identifiers. Only use fetch_file for a filename that "
    "appeared in an earlier search result — never guess or invent a filename. "
    "Do not repeat a search you have already run. As soon as any tool returns "
    "matches, stop calling tools and answer from what you have; do not keep "
    "searching for a more perfect source."
)

_ANSWER_SYSTEM = """\
You are a helpful assistant answering questions about a developer's knowledge base.
Answer based solely on the provided sources. If they do not contain enough
information to answer fully, say so explicitly rather than guessing.
Be concise and precise. Use markdown formatting where appropriate.
"""


class SynthesisAgent(Agent):
    def __init__(self, llm: LLMClient, store: VectorStore) -> None:
        self._store = store
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
            max_steps=_MAX_STEPS,
        )

    def _seed(
        self, task: str, state: AgentRunState
    ) -> Generator[Invocation, None, None]:
        seen = {chunk.id for chunk in state.chunks}
        before = len(state.chunks)
        for query in expand_query(task, self._llm, _SEED_EXTRA_QUERIES):
            for chunk in retrieve(
                query, self._llm, self._store, _SEED_TOP_K, _SEED_MIN_SCORE
            ):
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    state.chunks.append(chunk)

        if len(state.chunks) > before:
            invocation = Invocation(kind="tool", name="retrieve")
            state.usages.append(invocation)
            yield invocation
