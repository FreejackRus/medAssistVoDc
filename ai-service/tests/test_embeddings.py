from __future__ import annotations

from unittest.mock import Mock

from src.rag.embeddings import OllamaEmbeddings


def test_nomic_uses_retrieval_prefixes() -> None:
    embeddings = OllamaEmbeddings(model="nomic-embed-text-v2-moe")
    embeddings._client = Mock()
    embeddings._client.embed.side_effect = [
        {"embeddings": [[1.0], [2.0]]},
        {"embeddings": [[3.0]]},
    ]

    assert embeddings.embed(["Документ 1", "Документ 2"]) == [[1.0], [2.0]]
    assert embeddings.embed_query("Вопрос") == [3.0]

    assert embeddings._client.embed.call_args_list[0].kwargs["input"] == [
        "search_document: Документ 1",
        "search_document: Документ 2",
    ]
    assert (
        embeddings._client.embed.call_args_list[1].kwargs["input"]
        == "search_query: Вопрос"
    )


def test_other_embedding_models_are_not_prefixed() -> None:
    embeddings = OllamaEmbeddings(model="other-model")
    assert embeddings._document_input("text") == "text"
    assert embeddings._query_input("query") == "query"
