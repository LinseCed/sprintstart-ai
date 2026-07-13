# pyright: reportPrivateUsage=false
from api.dependencies import get_onboarding_orchestrator
from rag.retriever import get_bm25_cache
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def test_onboarding_orchestrator_shares_the_process_wide_bm25_cache() -> None:
    """Regression test for issue #129 #8: chat/agent retrieval and onboarding
    generation must tokenize the corpus once, not maintain independent caches.
    """
    orchestrator = get_onboarding_orchestrator(
        llm=StubLLMClient(), store=StubVectorStore()
    )

    assert orchestrator._pipeline._bm25_cache is get_bm25_cache()
