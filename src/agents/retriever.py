"""Retriever agent: for each job requirement, fetch the candidate's strongest evidence."""
from __future__ import annotations

from .. import config
from ..ingestion.store import KnowledgeBase
from ..schemas import EvidenceChunk, EvidencePack, JobProfile


def retrieve_evidence(job: JobProfile, kb: KnowledgeBase) -> EvidencePack:
    pack = EvidencePack()
    seen: dict[tuple[str, str], str] = {}

    requirements = sorted(job.requirements, key=lambda r: -r.importance)[: config.MAX_REQUIREMENTS]
    queries = [(r.name, r.name) for r in requirements]
    queries.append((f"__role__:{job.title}", job.title))  # general fit for the role itself

    for label, query in queries:
        ids: list[str] = []
        for text, source, score in kb.search(query, k=config.K_PER_REQUIREMENT):
            if score < config.MIN_RELEVANCE:
                continue
            key = (source, text)
            if key not in seen:
                cid = f"E{len(seen) + 1}"
                seen[key] = cid
                pack.chunks[cid] = EvidenceChunk(id=cid, text=text, source=source, score=score)
            ids.append(seen[key])
        pack.by_requirement[label] = ids
    return pack
