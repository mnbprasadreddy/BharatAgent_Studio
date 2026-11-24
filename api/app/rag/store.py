"""Reads and writes chunks in Postgres. Every chunk is tagged with a scope (which saved graph it
belongs to, or "inline") and the id of the source node it came from; search only ever looks at the
(scope, source ids) it is given, so two graphs that both use "s1" never see each other's chunks."""

from dataclasses import dataclass

import numpy as np

from ..db import get_conn

INLINE = "inline"


def graph_scope(graph_id: int | None) -> str:
    return f"graph:{graph_id}" if graph_id is not None else INLINE


@dataclass
class Chunk:
    id: int
    source_id: str
    file_name: str
    chunk_index: int
    page: int | None
    content: str
    similarity: float  # cosine similarity to the question, 1.0 = identical
    origin: str | None = None  # URL of the GitHub file or web page, if any


def _lock_source(conn, scope: str, source_id: str) -> None:
    """Writers of one source (in one scope) take turns. Without this, two ingestions of the same source at
    the same time both delete "the old chunks" before either commits, and both inserts survive (duplicates).
    The lock is released when the transaction ends."""
    conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{scope}|{source_id}",))


def replace_source(source_id: str, source_type: str, file_name: str, chunks: list[tuple],
                   vectors: list[list[float]], scope: str = INLINE, source_ref: int | None = None,
                   on_commit=None) -> int:
    """Delete the source's old chunks in this scope and insert the new ones in one transaction.
    chunks: [(page, text)] or [(page, text, file_name, origin)] when one source holds many files."""
    rows = []
    for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
        page, text = chunk[0], chunk[1]
        name = chunk[2] if len(chunk) > 2 else file_name
        origin = chunk[3] if len(chunk) > 3 else None
        rows.append((scope, source_id, source_type, name, i, page, text, origin, np.array(vec, dtype=np.float32), source_ref))
    with get_conn() as conn:
        _lock_source(conn, scope, source_id)
        conn.execute("DELETE FROM chunks WHERE scope = %s AND source_id = %s", (scope, source_id))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (scope, source_id, source_type, file_name, chunk_index, page, content, origin, embedding,"
                " source_ref) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)", rows)
        if on_commit is not None:  # e.g. mark the registry source ready in the same transaction
            on_commit(conn, len(rows))
    return len(rows)


def search(question_vector: list[float], source_ids: list[str], k: int, scope: str = INLINE) -> list[Chunk]:
    """The k chunks closest to the question, from the given sources in this scope only.
    The MATERIALIZED CTE filters first and then ranks exactly, so the HNSW index can never drop
    connected chunks or let a chunk from an unconnected source (or another graph) in."""
    if not source_ids:
        return []
    vec = np.array(question_vector, dtype=np.float32)
    with get_conn() as conn:
        rows = conn.execute(
            """
            WITH connected AS MATERIALIZED (
                SELECT id, source_id, file_name, chunk_index, page, content, origin, embedding
                FROM chunks WHERE scope = %s AND source_id = ANY(%s)
            )
            SELECT id, source_id, file_name, chunk_index, page, content, 1 - (embedding <=> %s) AS similarity, origin
            FROM connected ORDER BY embedding <=> %s LIMIT %s
            """,
            (scope, source_ids, vec, vec, k),
        ).fetchall()
    return [Chunk(*r[:6], similarity=float(r[6]), origin=r[7]) for r in rows]


def source_titles(scope: str, source_ids: list[str], limit: int = 5) -> list[str]:
    """Titles of the registry sources behind these chunk source ids, in this scope only (the sources this run
    may search). Used to give a vague first question ("what are its features?") the project's subject."""
    if not source_ids:
        return []
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT s.title FROM sources s
               WHERE s.status = 'ready' AND s.id IN (SELECT DISTINCT source_ref FROM chunks
                                                       WHERE scope = %s AND source_id = ANY(%s) AND source_ref IS NOT NULL)
               ORDER BY s.id LIMIT %s""",
            (scope, source_ids, limit)).fetchall()
    return [r[0] for r in rows]


def fetch(question_vector: list[float], scope: str, keys: list[tuple[str, int]]) -> list[Chunk]:
    """Chunks named by (source id, chunk index), re-read from the database now, with their similarity to the
    question. Used to reuse the evidence of an earlier answer: a chunk whose source was deleted, re-indexed or
    is not in this scope simply is not found."""
    if not keys:
        return []
    vec = np.array(question_vector, dtype=np.float32)
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, source_id, file_name, chunk_index, page, content, 1 - (embedding <=> %s) AS similarity, origin
            FROM chunks
            WHERE scope = %s AND (source_id, chunk_index) IN (SELECT * FROM unnest(%s::text[], %s::int[]))
            """,
            (vec, scope, [k[0] for k in keys], [k[1] for k in keys]),
        ).fetchall()
    return [Chunk(*r[:6], similarity=float(r[6]), origin=r[7]) for r in rows]


def chunk_vectors(chunk_ids: list[int]) -> dict[int, list[float]]:
    """Stored embeddings of the given chunks, for checking claims against them without re-embedding."""
    if not chunk_ids:
        return {}
    with get_conn() as conn:
        rows = conn.execute("SELECT id, embedding FROM chunks WHERE id = ANY(%s)", (chunk_ids,)).fetchall()
    return {r[0]: [float(x) for x in r[1]] for r in rows}


def source_info(source_id: str, scope: str = INLINE) -> dict:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT count(*), count(DISTINCT file_name), min(file_name), max(source_type), max(created_at)
               FROM chunks WHERE scope = %s AND source_id = %s""",
            (scope, source_id),
        ).fetchone()
    return {"source_id": source_id, "scope": scope, "chunks": row[0], "files": row[1], "file": row[2],
            "source_type": row[3], "ingested_at": row[4].isoformat() if row[4] else None}


def list_sources(scope: str = INLINE) -> list[dict]:
    """Every source that holds chunks in this scope, with its chunk count and file names (newest first),
    so the canvas can show what each node holds after a save, a reload or an upload made through the API."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT source_id, max(source_type), count(*), count(DISTINCT file_name),
                      (array_agg(DISTINCT file_name ORDER BY file_name))[1:20], max(created_at)
               FROM chunks WHERE scope = %s GROUP BY source_id ORDER BY source_id""",
            (scope,),
        ).fetchall()
    return [{"source_id": r[0], "scope": scope, "source_type": r[1], "chunks": r[2], "files": r[3],
             "file": r[4][0] if r[4] else None, "file_names": list(r[4] or []),
             "ingested_at": r[5].isoformat() if r[5] else None} for r in rows]


def clear_source(source_id: str, scope: str = INLINE) -> int:
    with get_conn() as conn:
        _lock_source(conn, scope, source_id)
        return conn.execute("DELETE FROM chunks WHERE scope = %s AND source_id = %s", (scope, source_id)).rowcount
