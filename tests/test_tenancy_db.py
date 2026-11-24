"""Phase 8 accounts, workspaces, projects, invitations, roles, tenant and project isolation, private
conversations, revocation, local-only projects, file authorization and the legacy migration, against a real
Postgres. Every person is a separately registered account with its own session cookie (tests/accounts.py).
Models and embeddings are faked. Skipped unless TEST_DATABASE_URL is set."""

import os
import threading
from pathlib import Path

import pytest

from accounts import PASSWORD, Account, agent_graph, fake_embeddings
from pdfgen import HOSTEL_RULES, REFUND_POLICY, make_pdf

DB_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = [pytest.mark.skipif(not DB_URL, reason="set TEST_DATABASE_URL to a Postgres with pgvector"),
              pytest.mark.real_usage]
if DB_URL:
    os.environ["DATABASE_URL"] = DB_URL


@pytest.fixture(scope="module")
def app():
    from fastapi.testclient import TestClient
    from app.main import app as fastapi_app
    with TestClient(fastapi_app):
        yield fastapi_app


@pytest.fixture(autouse=True)
def embeddings(monkeypatch):
    """Fake bge-m3, and every retrieved chunk counts as relevant (the fake vectors are coarse), so answers
    really go through the model; what is searched is still decided by the access rules under test."""
    import dataclasses
    from app.rag import retrieve
    fake_embeddings(monkeypatch)
    monkeypatch.setattr(retrieve, "settings", dataclasses.replace(retrieve.settings, min_similarity=0.0))


def sql(query, *args):
    from app.db import get_conn
    with get_conn() as conn:
        cur = conn.execute(query, args)
        return cur.fetchall() if cur.description else None


def company(owner, name="Bharat Technologies"):
    r = owner.post("/orgs", json={"name": name})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def join(owner, org_id, person, role="member"):
    """The real invitation flow: invite, accept with the invited account, owner confirms."""
    token = owner.post(f"/orgs/{org_id}/invitations", json={"email": person.email, "role": role}).json()["token"]
    assert person.post("/invitations/accept", json={"token": token}).json()["status"] == "pending"
    assert owner.post(f"/orgs/{org_id}/members/{person.id}/confirm").status_code == 200


def project_in(owner, org_id, name="E-commerce Platform", **extra):
    r = owner.post("/projects", json={"org_id": org_id, "name": name, **extra})
    assert r.status_code == 201, r.text
    return r.json()["id"]


@pytest.fixture(scope="module")
def world(app):
    """Owner A with a company and project; B contributor; C viewer; D outsider; E member of another project."""
    a, b, c, d, e = (Account(app, n) for n in ("owner_a", "contrib_b", "viewer_c", "outsider_d", "other_e"))
    org = company(a)
    for p in (b, c, e):
        join(a, org, p)
    proj = project_in(a, org)
    other = project_in(a, org, "HR Knowledge Base")
    assert a.put(f"/projects/{proj}/members/{b.id}", json={"role": "contributor"}).status_code == 200
    assert a.put(f"/projects/{proj}/members/{c.id}", json={"role": "viewer"}).status_code == 200
    assert a.put(f"/projects/{other}/members/{e.id}", json={"role": "contributor"}).status_code == 200
    src = b.add_pdf(proj, "architecture.pdf", make_pdf(REFUND_POLICY)).json()
    graph = a.post("/graphs", json={"project_id": proj, "name": "Shared agent",
                                    "graph": agent_graph({"s1": src["id"]})}).json()["id"]
    conv = c.post("/conversations", json={"project_id": proj, "graph_id": graph}).json()["id"]
    return dict(a=a, b=b, c=c, d=d, e=e, org=org, proj=proj, other=other, src=src["id"], graph=graph, conv=conv)


# ---------------- authentication ----------------
def test_registration_hashes_the_password_and_creates_a_personal_workspace(app):
    p = Account(app, "reg_check")
    stored = sql("SELECT password_hash, status FROM users WHERE id = %s", p.id)[0]
    assert stored[0].startswith("$argon2id$") and PASSWORD not in stored[0] and stored[1] == "active"
    assert "password" not in str(p.me) and p.me["workspaces"][0]["kind"] == "personal"
    assert p.get(f"/projects/{p.project}").json()["can_manage"] is True


def test_registration_validation_and_duplicates(app):
    from fastapi.testclient import TestClient
    c = TestClient(app)
    p = Account(app, "dup_check")
    body = {"full_name": "X", "email": p.email.upper(), "password": PASSWORD, "confirm_password": PASSWORD}
    assert c.post("/auth/register", json=body).status_code == 409  # emails are compared case-insensitively
    assert c.post("/auth/register", json={**body, "email": "not-an-email"}).status_code == 422
    assert c.post("/auth/register", json={**body, "email": "new1@example.test", "password": "short1"}).status_code == 422
    assert c.post("/auth/register", json={**body, "email": "new2@example.test", "confirm_password": "x"}).status_code == 422


def test_simultaneous_registrations_with_one_email_create_one_account(app):
    from fastapi.testclient import TestClient
    import uuid
    email = f"race-{uuid.uuid4().hex[:8]}@example.test"
    codes = []

    def go():
        codes.append(TestClient(app).post("/auth/register", json={"full_name": "Race", "email": email,
                                                                  "password": PASSWORD, "confirm_password": PASSWORD}).status_code)
    threads = [threading.Thread(target=go) for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(codes).count(201) == 1 and set(codes) <= {201, 409}
    assert sql("SELECT count(*) FROM users WHERE lower(email) = %s", email)[0][0] == 1


def test_login_errors_are_generic_and_rate_limited(app):
    from fastapi.testclient import TestClient
    from app.auth import login_limiter
    p = Account(app, "login_check")
    c = TestClient(app)
    wrong = c.post("/auth/login", json={"email": p.email, "password": "Wrong-password-1"})
    unknown = c.post("/auth/login", json={"email": "nobody@example.test", "password": "Wrong-password-1"})
    assert wrong.status_code == unknown.status_code == 401 and wrong.json() == unknown.json()
    for _ in range(10):
        c.post("/auth/login", json={"email": p.email, "password": "Wrong-password-1"})
    assert c.post("/auth/login", json={"email": p.email, "password": PASSWORD}).status_code == 429
    login_limiter.fails.clear()
    assert c.post("/auth/login", json={"email": p.email, "password": PASSWORD}).status_code == 200


def test_csrf_logout_revocation_and_expiry(app):
    from fastapi.testclient import TestClient
    p = Account(app, "session_check")
    token = p.c.cookies.get("ba_session")
    no_csrf = TestClient(app)
    no_csrf.cookies.set("ba_session", token)
    assert no_csrf.get("/auth/me").status_code == 200
    assert no_csrf.post("/orgs", json={"name": "x"}).status_code == 403  # a state change needs the CSRF header
    assert p.post("/auth/logout").status_code == 200
    assert no_csrf.get("/auth/me").status_code == 401  # the same cookie no longer works anywhere
    q = Account(app, "expiry_check")
    sql("UPDATE sessions SET expires_at = now() - interval '1 minute' WHERE user_id = %s", q.id)
    assert q.get("/auth/me").status_code == 401


def test_signing_in_again_replaces_the_old_session(app):
    p = Account(app, "rotate_check")
    old = p.c.cookies.get("ba_session")
    assert p.post("/auth/login", json={"email": p.email, "password": PASSWORD}).status_code == 200
    assert p.c.cookies.get("ba_session") != old
    from fastapi.testclient import TestClient
    stale = TestClient(app)
    stale.cookies.set("ba_session", old)
    assert stale.get("/auth/me").status_code == 401


def test_legacy_users_cannot_sign_in_and_unauthenticated_calls_are_refused(app):
    from fastapi.testclient import TestClient
    c = TestClient(app)
    assert sql("SELECT status FROM users WHERE name = 'demo'")[0][0] == "legacy"
    for method, path in [("get", "/orgs"), ("get", "/projects?org_id=1"), ("get", "/graphs/1"), ("get", "/sources/1"),
                         ("post", "/pipeline/run"), ("get", "/conversations?project_id=1"), ("get", "/usage/totals"),
                         ("get", "/status"), ("get", "/models"), ("get", "/tools"), ("get", "/users")]:
        assert getattr(c, method)(path).status_code in (401, 404, 405), path
    assert c.get("/health").json() == {"ok": True}


# ---------------- invitations and roles ----------------
def test_invitation_rules(app):
    a, b, x = Account(app, "inv_owner"), Account(app, "inv_bob"), Account(app, "inv_stranger")
    org = company(a, "Invite Co")
    made = a.post(f"/orgs/{org}/invitations", json={"email": b.email, "role": "member"}).json()
    preview = x.c.__class__(app).get(f"/invitations/preview?token={made['token']}").json()
    assert preview["organization"] == "Invite Co" and preview["email_hint"].startswith(b.email[0]) and b.email not in str(preview)
    assert x.post("/invitations/accept", json={"token": made["token"]}).status_code == 403  # another account
    assert b.post("/invitations/accept", json={"token": "wrong"}).status_code == 404
    assert b.post("/invitations/accept", json={"token": made["token"]}).json()["status"] == "pending"
    assert b.get(f"/orgs/{org}").status_code == 404  # waits for confirmation
    assert b.post("/invitations/accept", json={"token": made["token"]}).status_code == 409  # single use
    a.post(f"/orgs/{org}/members/{b.id}/confirm")
    assert b.get(f"/orgs/{org}").json()["role"] == "member"
    c = Account(app, "inv_carol")
    t2 = a.post(f"/orgs/{org}/invitations", json={"email": c.email}).json()
    a.post(f"/invitations/{t2['id']}/revoke")
    assert c.post("/invitations/accept", json={"token": t2["token"]}).status_code == 409
    t3 = a.post(f"/invitations/{t2['id']}/regenerate").json()
    sql("UPDATE invitations SET expires_at = now() - interval '1 hour' WHERE id = %s", t2["id"])
    assert c.post("/invitations/accept", json={"token": t3["token"]}).status_code == 410
    assert b.post(f"/orgs/{org}/invitations", json={"email": "z@example.test"}).status_code == 403  # members cannot invite


def test_concurrent_acceptance_creates_one_membership(app):
    a, b = Account(app, "conc_owner"), Account(app, "conc_bob")
    org = company(a, "Concurrent Co")
    token = a.post(f"/orgs/{org}/invitations", json={"email": b.email}).json()["token"]
    codes = []
    threads = [threading.Thread(target=lambda: codes.append(b.post("/invitations/accept", json={"token": token}).status_code))
               for _ in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert codes.count(200) == 1 and set(codes) <= {200, 409}
    assert sql("SELECT count(*) FROM org_members WHERE org_id = %s AND user_id = %s", org, b.id)[0][0] == 1


def test_role_escalation_is_refused(app):
    a, adm, m = Account(app, "esc_owner"), Account(app, "esc_admin"), Account(app, "esc_member")
    org = company(a, "Escalation Co")
    join(a, org, adm, "admin")
    join(a, org, m)
    assert adm.patch(f"/orgs/{org}/members/{adm.id}", json={"role": "owner"}).status_code == 403
    assert adm.post(f"/orgs/{org}/invitations", json={"email": "x@example.test", "role": "admin"}).status_code == 403
    assert adm.delete(f"/orgs/{org}/members/{a.id}").status_code == 403
    assert adm.patch(f"/orgs/{org}/members/{a.id}", json={"role": "member"}).status_code == 403  # nor demote the owner
    assert m.patch(f"/orgs/{org}/members/{m.id}", json={"role": "admin"}).status_code == 403
    assert m.delete(f"/orgs/{org}/members/{adm.id}").status_code == 403
    assert a.patch(f"/orgs/{org}/members/{a.id}", json={"role": "member"}).status_code == 409  # last owner
    assert a.delete(f"/orgs/{org}/members/{a.id}").status_code == 409
    assert m.patch(f"/orgs/{org}", json={"name": "Hacked"}).status_code == 403
    assert adm.delete(f"/orgs/{org}/members/{m.id}").status_code == 200  # admins manage ordinary members


# ---------------- shared knowledge and isolation ----------------
def test_shared_knowledge_is_used_without_uploading_again(world):
    c = world["c"]
    assert [s["title"] for s in c.get(f"/projects/{world['proj']}/sources").json()] == ["architecture.pdf"]
    r, events = c.ask(question="refund policy", graph_id=world["graph"])
    assert r.status_code == 200 and events[-1]["type"] == "result"
    agent = next(e for e in events if e.get("stage") == "agent" and e.get("status") == "done")
    assert {x["file"] for x in agent["detail"]["retrieval"]["candidates"]} == {"architecture.pdf"}
    assert sql("SELECT count(*) FROM sources WHERE project_id = %s", world["proj"])[0][0] == 1  # one copy


def test_project_roles(world):
    b, c = world["b"], world["c"]
    assert c.add_pdf(world["proj"], "x.pdf", make_pdf(HOSTEL_RULES)).status_code == 403  # viewer cannot add
    assert c.post("/graphs", json={"project_id": world["proj"], "name": "x", "graph": agent_graph()}).status_code == 403
    assert c.post("/pipeline/run?stream=false", json={"question": "hi", "project_id": world["proj"],
                                                       "graph": agent_graph()}).status_code == 403  # unsaved graphs too
    g = world["a"].get(f"/graphs/{world['graph']}").json()
    edit = {"name": "changed", "graph": {"nodes": g["nodes"], "edges": g["edges"]}}
    assert b.put(f"/graphs/{world['graph']}", json=edit).status_code == 403  # not her agent
    assert b.put(f"/projects/{world['proj']}/members/{world['d'].id}", json={"role": "viewer"}).status_code == 403
    assert c.delete(f"/sources/{world['src']}").status_code == 403


def test_same_company_other_project_is_invisible(world):
    e, proj = world["e"], world["proj"]
    for path in [f"/projects/{proj}", f"/projects/{proj}/sources", f"/graphs?project_id={proj}", f"/graphs/{world['graph']}",
                 f"/sources/{world['src']}", f"/sources/{world['src']}/file", f"/conversations?project_id={proj}",
                 f"/usage/totals?project_id={proj}"]:
        assert e.get(path).status_code == 404, path
    assert e.ask(question="refund", graph_id=world["graph"])[0].status_code == 404
    assert e.ask(question="refund", project_id=proj)[0].status_code == 404
    assert world["proj"] not in [p["id"] for p in e.get(f"/projects?org_id={world['org']}").json()]


def test_outsider_is_refused_on_every_route(world):
    d, proj, src, graph, conv, org = world["d"], world["proj"], world["src"], world["graph"], world["conv"], world["org"]
    gets = [f"/orgs/{org}", f"/orgs/{org}/members", f"/orgs/{org}/invitations", f"/projects/{proj}",
            f"/projects/{proj}/sources", f"/projects/{proj}/members", f"/graphs?project_id={proj}", f"/graphs/{graph}",
            f"/sources/{src}", f"/sources/{src}/file", f"/conversations?project_id={proj}", f"/conversations/{conv}",
            f"/usage/totals?org_id={org}", f"/usage/recent?project_id={proj}", f"/projects?org_id={org}"]
    for path in gets:
        assert d.get(path).status_code == 404, path
    writes = [d.add_pdf(proj, "x.pdf", make_pdf(HOSTEL_RULES)), d.post(f"/projects/{proj}/sources/url", json={"url": "https://example.com"}),
              d.delete(f"/sources/{src}"), d.post(f"/sources/{src}/reindex"), d.delete(f"/graphs/{graph}"),
              d.patch(f"/projects/{proj}", json={"name": "x"}), d.delete(f"/projects/{proj}"),
              d.post("/conversations", json={"project_id": proj}), d.patch(f"/conversations/{conv}", json={"title": "x"}),
              d.delete(f"/conversations/{conv}"), d.post(f"/orgs/{org}/invitations", json={"email": d.email}),
              d.post("/pipeline/validate?project_id=%d" % proj, json=agent_graph())]
    assert all(r.status_code == 404 for r in writes), [r.status_code for r in writes]
    for body in [dict(graph_id=graph), dict(project_id=proj), dict(conversation_id=conv),
                 dict(project_id=d.project, graph=agent_graph({"s1": src})),         # someone else's source in my graph
                 dict(graph_id=graph, graph=agent_graph({"s1": src}))]:
        r, _ = d.ask(question="refund policy", **body)
        assert r.status_code in (404, 422), (body, r.status_code)
    assert d.post("/graphs", json={"project_id": d.project, "name": "steal", "graph": agent_graph({"s1": src})}).status_code == 422


def test_conversations_are_private_and_keep_context(world, monkeypatch):
    from app.models import gemini
    prompts = []

    def stream(system, prompt, **kw):
        prompts.append(prompt)
        yield {"text": "80 percent of the test amount is refunded [1]."}
        yield {"usage": {"input_tokens": 10, "output_tokens": 5}}
    monkeypatch.setattr(gemini, "stream", stream)
    b, c, a = world["b"], world["c"], world["a"]
    conv_b = b.post("/conversations", json={"project_id": world["proj"], "graph_id": world["graph"]}).json()["id"]
    r, _ = c.ask(question="How much is refunded for a lab test?", conversation_id=world["conv"], request_id="q1")
    assert r.status_code == 200
    r, _ = c.ask(question="And for the consultation?", conversation_id=world["conv"], request_id="q2")
    assert "How much is refunded for a lab test?" in prompts[-1]  # the follow-up sees the earlier question
    assert c.ask(question="again", conversation_id=world["conv"], request_id="q2")[0].status_code == 409  # no duplicates
    msgs = c.get(f"/conversations/{world['conv']}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert msgs[1]["model"].startswith("gemini:") and msgs[1]["citations"] and msgs[1]["input_tokens"] == 10
    assert b.get(f"/conversations/{world['conv']}").status_code == 404   # a colleague cannot read it
    assert a.get(f"/conversations/{world['conv']}").status_code == 404   # neither can the owner
    assert [x["id"] for x in b.get(f"/conversations?project_id={world['proj']}").json()] == [conv_b]
    assert b.ask(question="x", conversation_id=world["conv"])[0].status_code == 404


def test_failed_answer_is_stored_as_an_error_not_an_answer(world, monkeypatch):
    from app.models import gemini, ollama
    from app.models.base import ProviderError

    def broken(system, prompt, **kwargs):
        raise ProviderError("down", code="network_error")
        yield  # pragma: no cover
    monkeypatch.setattr(gemini, "stream", broken)
    monkeypatch.setattr(ollama, "stream", broken)
    c = world["c"]
    conv = c.post("/conversations", json={"project_id": world["proj"], "graph_id": world["graph"]}).json()["id"]
    r, events = c.ask(question="refund policy", conversation_id=conv)
    assert events[-1]["type"] == "error"
    msgs = c.get(f"/conversations/{conv}").json()["messages"]
    assert msgs[-1]["role"] == "assistant" and msgs[-1]["status"] == "error"


def test_usage_ownership(world):
    a, c = world["a"], world["c"]
    c.ask(question="refund policy", graph_id=world["graph"])
    row = sql("SELECT user_id, org_id, project_id, graph_id FROM usage ORDER BY id DESC LIMIT 1")[0]
    assert row == (c.id, world["org"], world["proj"], world["graph"])
    assert a.get(f"/usage/totals?org_id={world['org']}").json()["everyone"] is True
    mine = c.get(f"/usage/totals?org_id={world['org']}").json()
    assert mine["everyone"] is False and mine["requests"] == len(sql("SELECT 1 FROM usage WHERE user_id = %s AND org_id = %s", c.id, world["org"]))
    assert "content" not in str(a.get(f"/usage/recent?org_id={world['org']}").json())


def test_files_are_served_only_to_project_members(world):
    r = world["c"].get(f"/sources/{world['src']}/file")
    assert r.status_code == 200 and r.content.startswith(b"%PDF-") and r.headers["cache-control"] == "private, no-store"
    assert world["d"].get(f"/sources/{world['src']}/file").status_code == 404


def test_local_only_project_never_reaches_a_cloud_model(world, monkeypatch):
    from app.models import gemini, groq, ollama
    cloud = []
    monkeypatch.setattr(gemini, "stream", lambda *a, **k: cloud.append("gemini") or iter(()))
    monkeypatch.setattr(groq, "stream", lambda *a, **k: cloud.append("groq") or iter(()))

    def local(system, prompt, **kwargs):
        yield {"text": '{"calls": []}' if "JSON" in system else "80 percent of the test amount is refunded [1]."}
        yield {"usage": {"input_tokens": 5, "output_tokens": 5}}
    monkeypatch.setattr(ollama, "stream", local)
    a = world["a"]
    p = project_in(a, world["org"], "Confidential", local_only=True)
    sid = a.add_pdf(p, "secret.pdf", make_pdf(REFUND_POLICY)).json()["id"]
    calc = {"id": "t1", "type": "tool_calculator", "config": {}}
    gid = a.post("/graphs", json={"project_id": p, "name": "cloud agent", "graph": agent_graph({"s1": sid}, [calc], model="groq")}).json()["id"]
    assert a.ask(question="refund", graph_id=gid)[0].status_code == 400            # the agent's cloud model
    assert a.ask(question="refund", graph_id=gid, model="gemini")[0].status_code == 400
    assert a.ask(question="refund", project_id=p, model="groq")[0].status_code == 400
    r, events = a.ask(question="My lab test cost Rs. 1200, how much do I get back?", graph_id=gid, model="local")
    assert r.status_code == 200 and events[-1]["local_only"] is True and events[-1]["model"].startswith("local")
    assert cloud == []


def test_removing_access_takes_effect_immediately(world, app):
    a, b = world["a"], Account(app, "revoked_f")
    join(a, world["org"], b)
    a.put(f"/projects/{world['proj']}/members/{b.id}", json={"role": "viewer"})
    conv = b.post("/conversations", json={"project_id": world["proj"]}).json()["id"]
    assert b.get(f"/projects/{world['proj']}").status_code == 200
    a.delete(f"/projects/{world['proj']}/members/{b.id}")   # b is still signed in
    for path in [f"/projects/{world['proj']}", f"/projects/{world['proj']}/sources", f"/conversations/{conv}",
                 f"/sources/{world['src']}/file", f"/graphs/{world['graph']}"]:
        assert b.get(path).status_code == 404, path
    assert b.ask(question="refund", graph_id=world["graph"])[0].status_code == 404
    assert b.ask(question="refund", conversation_id=conv)[0].status_code == 404
    a.put(f"/projects/{world['proj']}/members/{b.id}", json={"role": "viewer"})
    a.delete(f"/orgs/{world['org']}/members/{b.id}")        # removed from the company: all project roles go too
    assert b.get(f"/projects/{world['proj']}").status_code == 404
    assert sql("SELECT count(*) FROM project_members WHERE user_id = %s", b.id)[0][0] == 0


KNOWLEDGE = Path(__file__).resolve().parent.parent / "samples" / "knowledge"
FILE_FACTS = {"warehouse-notes.txt": "6:30 PM every weekday", "support-runbook.md": "within 15 minutes",
              "leave-policy.docx": "18 days of paid leave", "courier-partners.csv": "Max parcel weight (kg): 32",
              "gift-cards.xlsx": "Validity (months): 9"}


def test_every_file_type_is_extracted_chunked_embedded_stored_and_shared(world):
    """TXT, Markdown, DOCX, CSV and XLSX (PDF is covered above): uploaded once by a contributor, read by a viewer."""
    b, c, d, proj = world["b"], world["c"], world["d"], world["proj"]
    for name, fact in FILE_FACTS.items():
        data = (KNOWLEDGE / name).read_bytes()
        r = b.add_pdf(proj, name, data)
        assert r.status_code == 201, (name, r.text)
        sid = r.json()["id"]
        assert r.json()["status"] == "ready" and r.json()["chunk_count"] >= 1, name
        stored = sql("SELECT count(*), count(embedding), string_agg(content, ' ') FROM chunks WHERE source_ref = %s", sid)[0]
        assert stored[0] == stored[1] == r.json()["chunk_count"] and fact in stored[2], (name, stored[:2])
        assert sid in [s["id"] for s in c.get(f"/projects/{proj}/sources").json()]          # the viewer sees it
        got = c.get(f"/sources/{sid}/file")
        assert got.status_code == 200 and got.content == data, name                         # same bytes back
        assert d.get(f"/sources/{sid}").status_code == d.get(f"/sources/{sid}/file").status_code == 404


def test_agents_reuse_project_chunks_without_embedding_again(world):
    a, src, proj = world["a"], world["src"], world["proj"]
    before = sql("SELECT array_agg(id ORDER BY id) FROM chunks WHERE source_ref = %s", src)[0][0]
    for name in ("second agent", "third agent"):
        assert a.post("/graphs", json={"project_id": proj, "name": name, "graph": agent_graph({"s1": src})}).status_code == 201
    world["c"].ask(question="refund policy", graph_id=world["graph"])
    world["c"].ask(question="refund policy", project_id=proj)
    assert sql("SELECT array_agg(id ORDER BY id) FROM chunks WHERE source_ref = %s", src)[0][0] == before  # same rows
    assert sql("SELECT count(*) FROM agent_sources WHERE source_id = %s", src)[0][0] == 3


def _revoking_model(monkeypatch, world, tokens: int):
    """A model that removes the viewer from the project after its first token, while the answer is streaming."""
    from app.models import gemini

    def stream(system, prompt, **kw):
        yield {"text": "80 percent "}
        sql("DELETE FROM project_members WHERE project_id = %s AND user_id = %s", world["proj"], world["c"].id)
        for _ in range(tokens):
            yield {"text": "of the test amount "}
        yield {"text": "is refunded [1]."}
        yield {"usage": {"input_tokens": 10, "output_tokens": 5}}
    monkeypatch.setattr(gemini, "stream", stream)


def test_access_removed_while_an_answer_streams_stops_it(world, monkeypatch):
    from app.routes import pipeline
    monkeypatch.setattr(pipeline, "RECHECK_SECONDS", 0.0)   # re-check on every event (normally every 2 s)
    _revoking_model(monkeypatch, world, tokens=5)
    c = world["c"]
    conv = c.post("/conversations", json={"project_id": world["proj"], "graph_id": world["graph"]}).json()["id"]
    try:
        r, events = c.ask(question="How much is refunded for a lab test?", conversation_id=conv)
        assert events[-1]["type"] == "error" and events[-1]["code"] == "access_revoked"
        assert not any(e["type"] == "result" for e in events)
        assert not sql("SELECT 1 FROM messages WHERE conversation_id = %s AND role = 'assistant' AND status = 'ok'", conv)
    finally:
        world["a"].put(f"/projects/{world['proj']}/members/{c.id}", json={"role": "viewer"})


def test_final_answer_is_withheld_if_access_was_removed_during_the_run(world, monkeypatch):
    """Even before the 2 s timer fires, the final result (answer and citations) is never sent after a removal."""
    _revoking_model(monkeypatch, world, tokens=0)
    try:
        r, events = world["c"].ask(question="How much is refunded for a lab test?", graph_id=world["graph"])
        assert events[-1]["code"] == "access_revoked" and not any(e["type"] == "result" for e in events)
    finally:
        world["a"].put(f"/projects/{world['proj']}/members/{world['c'].id}", json={"role": "viewer"})


def _rewriting_model(monkeypatch, module, answer, calls):
    """A model that answers the rewrite prompt with a query and other prompts with `answer`; records which."""
    from app.rag import conversation

    def stream(system, prompt, **kw):
        is_rewrite = system == conversation.REWRITE_SYSTEM
        calls.append(f"{module.__name__.rsplit('.', 1)[-1]}:{'rewrite' if is_rewrite else 'answer'}")
        yield {"text": "Detailed explanation of the refund policy" if is_rewrite else answer}
        yield {"usage": {"input_tokens": 10, "output_tokens": 5}}
    monkeypatch.setattr(module, "stream", stream)


def test_follow_up_rewrite_in_a_local_only_project_never_reaches_the_cloud(world, monkeypatch):
    from app.models import gemini, groq, ollama
    calls = []
    for cloud in (gemini, groq):
        monkeypatch.setattr(cloud, "stream", lambda *a, **k: calls.append("CLOUD") or iter(()))
    _rewriting_model(monkeypatch, ollama, "80 percent of the test amount is refunded [1].", calls)
    a = world["a"]
    p = project_in(a, world["org"], "Private follow-ups", local_only=True)
    a.add_pdf(p, "secret.pdf", make_pdf(REFUND_POLICY))
    conv = a.post("/conversations", json={"project_id": p}).json()["id"]
    assert a.ask(question="How much is refunded for a lab test?", conversation_id=conv, model="local")[0].status_code == 200
    r, events = a.ask(question="Explain it in detail.", conversation_id=conv, model="local")
    assert r.status_code == 200 and "CLOUD" not in calls and "ollama:rewrite" in calls
    steps = [s["model"] for s in next(e for e in events if e.get("stage") == "agent" and e.get("status") == "done")
             ["detail"]["generation"]["steps"]]
    assert steps and all(m.startswith("local") for m in steps)


def test_a_deleted_source_is_never_reused_through_conversation_history(world, monkeypatch):
    from app.models import gemini
    calls = []
    _rewriting_model(monkeypatch, gemini, "The gate closes at 9:30 PM [1][2].", calls)
    a = world["a"]
    p = project_in(a, world["org"], "History and deletion")
    gone = a.add_pdf(p, "temporary.pdf", make_pdf(HOSTEL_RULES)).json()["id"]
    a.add_pdf(p, "kept.pdf", make_pdf(REFUND_POLICY))
    conv = a.post("/conversations", json={"project_id": p}).json()["id"]
    r, events = a.ask(question="What time does the hostel gate close?", conversation_id=conv, model="gemini")
    assert "temporary.pdf" in {c["file"] for c in events[-1]["citations"]}   # the first answer cited it
    assert a.delete(f"/sources/{gone}").status_code == 200
    r, events = a.ask(question="Explain it in detail.", conversation_id=conv, model="gemini")
    retrieval = next(e for e in events if e.get("stage") == "agent" and e.get("status") == "done")["detail"]["retrieval"]
    assert retrieval["follow_up"] is True and not any("temporary.pdf" in x for x in retrieval["reused"])
    assert all(c["file"] != "temporary.pdf" for c in retrieval["candidates"])
    assert all(c["file"] != "temporary.pdf" for c in events[-1].get("citations") or [])


def test_project_source_names_come_only_from_sources_this_run_may_search(world):
    """The names added to a vague first question are read from the sources being searched, in this project's
    scope only: another project's sources (or a deleted one) never leak into the search."""
    from app.rag import store
    a = world["a"]
    p1, p2 = project_in(a, world["org"], "Names one"), project_in(a, world["org"], "Names two")
    keep = a.add_pdf(p1, "Visible Handbook.pdf", make_pdf(REFUND_POLICY)).json()["id"]
    gone = a.add_pdf(p1, "Removed Notes.pdf", make_pdf(HOSTEL_RULES)).json()["id"]
    other = a.add_pdf(p2, "Other Project Secret.pdf", make_pdf(HOSTEL_RULES)).json()["id"]
    a.delete(f"/sources/{gone}")
    ids = [f"src{keep}", f"src{gone}", f"src{other}"]
    assert store.source_titles(f"project:{p1}", ids) == ["Visible Handbook.pdf"]
    assert store.source_titles(f"project:{p2}", ids) == ["Other Project Secret.pdf"]


def test_project_chat_uses_every_ready_project_source(world):
    r, events = world["c"].ask(question="refund policy", project_id=world["proj"])
    agent = next(e for e in events if e.get("stage") == "agent" and e.get("status") == "done")
    ready = sql("SELECT id FROM sources WHERE project_id = %s AND status = 'ready' ORDER BY id", world["proj"])
    assert sorted(agent["detail"]["retrieval"]["searched"]) == sorted(f"source_{r[0]}" for r in ready)
    assert f"source_{world['src']}" in agent["detail"]["retrieval"]["searched"]


def test_a_source_interrupted_by_a_restart_is_marked_failed_not_left_processing(app):
    owner = Account(app, "restart")
    sid = sql("INSERT INTO sources (project_id, uploaded_by, kind, title, status) VALUES (%s, %s, 'pdf', 'cut.pdf', "
              "'processing') RETURNING id", owner.project, owner.id)[0][0]
    from app.db import get_conn, reset_interrupted_sources
    with get_conn() as conn:
        assert reset_interrupted_sources(conn) >= 1
    status, error = sql("SELECT status, error FROM sources WHERE id = %s", sid)[0]
    assert status == "failed" and "Interrupted" in error


# ---------------- migration ----------------
def test_migration_is_recorded_once_and_rerun_is_a_no_op(app):
    from app import migrations
    from app.config import settings
    assert migrations.run(settings.database_url, settings.upload_dir) == []
    assert sql("SELECT version FROM schema_migrations") == [("0001",)]


def test_legacy_data_is_quarantined_until_an_operator_assigns_it(app):
    from app.admin import claim_legacy
    legacy = sql("SELECT id FROM organizations WHERE kind = 'legacy'")[0][0]
    assert sql("SELECT count(*) FROM chunks WHERE source_ref IS NULL")[0][0] == 0
    legacy_graph = sql("SELECT g.id FROM graphs g JOIN projects p ON p.id = g.project_id WHERE p.org_id = %s LIMIT 1", legacy)
    p = Account(app, "legacy_claimer")
    if legacy_graph:
        assert p.get(f"/graphs/{legacy_graph[0][0]}").status_code == 404
    assert "now owns" in claim_legacy(p.email)
    assert p.get(f"/orgs/{legacy}").json()["role"] == "owner"
    if legacy_graph:
        assert p.get(f"/graphs/{legacy_graph[0][0]}").status_code == 200
    sql("DELETE FROM org_members WHERE org_id = %s AND user_id = %s", legacy, p.id)  # leave it quarantined
