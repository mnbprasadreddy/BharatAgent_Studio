# Phase 8 implementation: accounts, workspaces, projects and shared knowledge

Plan and decisions: [PHASE_8_PLAN.md](PHASE_8_PLAN.md). Security model and tests:
[PHASE_8_SECURITY.md](PHASE_8_SECURITY.md). Walkthrough: [PHASE_8_DEMO.md](PHASE_8_DEMO.md).

## What a user can do now

1. Register (full name, email, password) and land in a private **personal workspace** with a first project.
2. Create a **company** (they become its owner), **invite** colleagues by link, confirm them, give them roles.
3. Create **projects**; assign company members as **contributor** (adds knowledge, builds agents) or **viewer**.
4. Add **knowledge** once per project: PDF, Word (.docx), Markdown, TXT, CSV, Excel (.xlsx), public GitHub
   repositories and documentation pages. Status, uploader, size, revision and errors are real records.
5. Build **shared agents** in the visual builder from the project's existing sources and tools.
6. **Chat** with the project's knowledge or with a shared agent; conversations are saved, private, and follow-up
   questions keep context. Every answer shows the real model, citations, tool calls, tokens and cost.
7. See **usage**: their own, or (owners/admins) the workspace's per project, model and person.

## Authentication and sessions (`api/app/auth.py`, `api/app/routes/auth.py`)

- `POST /auth/register` → Argon2id hash, user (`status = active`), personal workspace (organization of kind
  `personal`, the user as owner), project "My first project", session.
- `POST /auth/login` → generic error, 10 failures per email+address per 10 minutes, new session (old one revoked).
- `POST /auth/logout` revokes the session; `POST /auth/password` changes the password and revokes all sessions.
- `GET /auth/me` → safe profile, workspaces (with role and status) and the CSRF token.
- Sessions: table `sessions` (SHA-256 of the token, expiry, revocation, last seen); cookie `ba_session`
  (HttpOnly, SameSite=Lax, Secure via `COOKIE_SECURE`). CSRF header `X-CSRF-Token` on every state change.

## Organizations, roles, invitations (`api/app/routes/orgs.py`)

- One `organizations` table: `kind` = `personal` | `company` | `legacy`; `org_members` (role owner/admin/member,
  status pending/active, unique per user and organization).
- Owners: settings, roles, invitations (admins and members), remove anyone but the last owner.
  Admins: invite members, confirm, remove ordinary members. Members: use their projects.
- Invitations: `POST /orgs/{id}/invitations` (token returned once), `GET` list, `revoke`, `regenerate`,
  public `GET /invitations/preview?token=` (company, role, masked email), `POST /invitations/accept` (matching
  email, single use, pending until `POST /orgs/{id}/members/{user}/confirm`).

## Projects (`api/app/routes/projects.py`)

`projects` (organization, name, description, `local_only`, archived) and `project_members` (contributor/viewer).
Owners and admins of the workspace have access to every project of it (documented policy); members only to
assigned ones. Endpoints: list per workspace, create, get (with counts and recent real activity), update
(name, description, local-only, archive), delete (cascade to sources, chunks, files, agents, conversations),
list/assign/remove members.

## Shared knowledge (`api/app/knowledge.py`, `api/app/routes/sources.py`, `api/app/rag/extract.py`)

- `sources` registry per project: kind, title, original name, URL, storage path, size, SHA-256, revision (GitHub
  commit), status pending → processing → ready / failed, chunk count, error, uploader, timestamps.
- Ingestion stays synchronous inside the request (honest status: the source is `ready` only in the same
  transaction that writes its chunks; a failed new source is `failed` with its error and keeps no file; a failed
  re-index keeps the previous working content and records the error). A source left `processing` by an api
  restart is marked `failed` ("Interrupted by an api restart") when the api starts, so nothing stays stuck.
  Each upload writes its own temporary file, so concurrent replacements of one source do not collide.
- Chunks: scope `project:<id>`, source id `src<id>`, `source_ref` foreign key with `ON DELETE CASCADE`; one copy per
  project, reused by every agent and member. No cross-project deduplication (it would reveal what other tenants hold).
- Files: `uploads/project_<id>/source_<id>/`; downloaded only through `GET /sources/{id}/file` after the access
  check; replaced, refused and deleted files are removed (fixes the Phase 7.1 finding).
- New formats: TXT, Markdown, DOCX (paragraph text from the document XML), CSV and XLSX (rows as
  "Header: value" lines, values only, limits in `extract.py`).

## Shared agents (`api/app/routes/graphs.py`, builder)

- `graphs` gains `project_id`, `created_by`, `description`, `created_at`; `agent_sources` mirrors which node uses
  which source. A source node holds `config.source_ref`; saving checks every reference belongs to the project.
- Viewers open and run agents; contributors create agents and edit their own; owners/admins edit all.
  Deleting an agent never deletes sources; deleting a source unlinks it from agents (they then refuse to run
  until a new source is picked).
- The builder lists the project's agents, picks existing project sources in the inspector (or adds a new one to
  the project), saves into the project, and runs unsaved edits only for editors.

## Execution and chat (`api/app/routes/pipeline.py`, `api/app/routes/conversations.py`)

- `POST /pipeline/run` takes `graph_id` (agent), `project_id` (project chat, or an editor's unsaved graph),
  `conversation_id` and `request_id`; the user, project and allowed sources come from the database.
- Project chat = a graph built by the api: the five blocks, every **ready** source of the project, the calculator.
- The engine (`run_pipeline`) gained three optional inputs: `source_map` (node → authorized stored source),
  `history` (last `CHAT_HISTORY_MESSAGES` messages, each cut to 600 characters; the question is always searched
  on its own, and a short one is also searched joined with the previous question, best chunks of both kept, so
  "what happens if it is down?" finds the payment rules and a new topic is not crowded out by the last one), and `owner` (organization, project, agent, conversation for usage).
  Retrieval, citations, claim checks, tools, fallback and local-only are unchanged.
- `conversations` (owner, project, agent, title) and `messages` (role, content, status ok/error, real model,
  requested model, fallback, citations, tool calls, tokens, cost, request id). The question is stored when the
  request starts; the answer when the `result` event arrives; a failure is stored as an `error` message, never as
  an answer; a repeated `request_id` is refused (no duplicate messages).

## Usage ownership (`api/app/routes/usage.py`, `api/app/usage.py`)

`usage` gains `org_id`, `project_id`, `graph_id`, `conversation_id`; the user is the session user. Endpoints take
`org_id` / `project_id` / `model` / `days`: members get their own rows; owners and admins get the workspace's,
including per project (`/usage/by-project`) and per person; never message text.

## Database relationships

```
users 1-n org_members n-1 organizations 1-n projects 1-n sources 1-n chunks (vector 1024)
users 1-n sessions        organizations 1-n invitations   projects 1-n project_members n-1 users
projects 1-n graphs 1-n agent_sources n-1 sources         projects 1-n conversations 1-n messages
usage n-1 users / organizations / projects / graphs / conversations (SET NULL on delete)
```
Foreign keys everywhere; unique constraints on emails (`lower(email)`), slugs, memberships, invitation token
hashes, session token hashes, `(conversation, request_id, role)`; CHECK constraints on every role and status.

## Migration (`api/app/migrations/`)

- New versioned migrations: `schema_migrations` table, one transaction per migration, advisory lock, applied at
  api start after the baseline `schema.sql`. `m0001_collaboration` adds everything above.
- Existing data is preserved and quarantined: one "Legacy workspace (imported)" with **no members** and two
  projects (imported agents, imported quick-chat sources). Every old chunk group became a registry source (PDF
  files found in the uploads volume and linked; GitHub revisions taken from the stored links); chunks were
  re-pointed without recomputing any vector; old agents' source nodes got their `source_ref`; usage rows were
  linked to the legacy workspace; old user names became `legacy` users that cannot sign in.
- Main database on 2026-11-24, before → after: users 20 → 20, agents 71 → 71, chunks 206 → 206 (206 vectors),
  usage 1,106 → 1,106; 82 sources created (53 PDFs with their files, 11 repos, 18 pages); 0 chunks without a
  source; 72 agent links. Tested first on a restored copy, twice (the second run applies nothing).
- Claim: `docker compose exec api python -m app.admin claim-legacy <email>`; status: `python -m app.admin status`.
- Rollback: restore `backups/phase8_baseline_20261124.dump` with `pg_restore --clean` and check out `e69e3f3`.

## Verification scripts

Every earlier verify script and the Python tools (`model_matrix.py`, `load_test.py`, `eval_languages.py`) sign in
as a fresh test account through the real registration (`scripts/_session.ps1`, `scripts/_session.py`) and use
project sources; nothing bypasses authentication. Checks of removed features were replaced, each with a comment:
per-graph uploads became project sources plus a refused cross-project source (422); the open user list became
"requests are made as the signed-in account"; a run whose source was deleted is refused before it runs
(422 `source_not_ready`) instead of answering "I don't know". `scripts/verify_all.ps1` runs them all.
`scripts/ui_journeys.py` is now the multi-user browser journey (owner, contributor, consumer, outsider in four
browser profiles) with the p8-* screenshots in docs/screenshots.

## Verification results (2026-11-24, latest run of each, real stack with the GPU override)

| Check | PASS | FAIL | SKIP | BLOCKED (Gemini daily free quota) |
|---|---|---|---|---|
| pytest (with the test database) | 310 | 0 | 0 | - |
| Vitest | 25 | 0 | 0 | - |
| ui_journeys.py (4 browser profiles, 4 screen sizes) | 36 | 0 | 0 | - |
| verify_phase1 | 22 | 0 | 0 | 0 |
| verify_phase2 | 29 | 0 | 0 | 0 |
| verify_phase3 (real Gemini) | 13 | 0 | 0 | 10 |
| verify_phase3 -UseFakeGemini (MOCKED model, not proof of Gemini) | 20 | 0 | 3 | 0 |
| verify_phase4 | 27 | 0 | 0 | 3 |
| verify_phase5 | 24 | 0 | 0 | 0 |
| verify_models | 17 | 0 | 0 | 1 |
| verify_phase6 | 22 | 0 | 0 | 1 |
| verify_phase7 (includes the browser journeys) | 21 | 0 | 0 | 1 |
| verify_phase7_1 | 43 | 0 | 0 | 0 |
| verify_phase8 | 35 | 0 | 0 | 1 |

Every BLOCKED check is a real Gemini request refused with HTTP 429 (daily free-tier quota for gemini-2.5-flash,
resets at midnight Pacific time); the local model answered instead and that is never counted as a Gemini pass.
Groq, Qwen 3 14B and Qwen 2.5 3B answered for real. Observed variance: asked "My order is Rs. 450. How much more do
I need to spend to get free delivery?", Groq answered Rs. 49 (correct, cited, calculator used) in 3 of 4 probes;
once it reasoned "above Rs. 499" to Rs. 500 and Rs. 50, the claim check removed that unsupported sentence and the
answer became "I don't know" (one earlier verify_phase8 run failed on this; the latest passed).

## Conversational retrieval fix (follow-up questions)

Reported in manual testing: after "What is this project about?" (cited answer), the follow-up "tell clerly about
this allobrate the explination" was answered "I don't know" without any model call.

Root cause, measured with the real bge-m3 index (threshold 0.50): the follow-up was searched as typed (best chunk
0.442) and glued to the previous question (0.411, worse); nothing cleared the threshold, so the agent abstained
before asking the model. Conversation history reached only the answer prompt, never the search. Broad questions
also relied on whatever single chunk cleared the threshold ("What is this project about?": one chunk at 0.501;
"What are its main features?": none, best 0.425). Two smaller defects on the same path: "Explain the same thing
in Telugu" was answered in English (the language came only from the question's script), and the answer prompt
always asked for one or two sentences.

Fix (`api/app/rag/conversation.py`, `api/app/pipeline/stages/agent.py`, `retrieve.py`, `store.py`, `prompt.py`,
`input.py`, `routes/pipeline.py`):
- A question that looks like a follow-up (refers back with it/this/that/its..., asks to explain, simplify, give an
  example, summarise, translate or continue, tolerates common misspellings, or is very short) is rewritten by the
  same model into one standalone search query, using the last few turns (bounded). Both the rewrite and the
  question are searched and merged. A standalone question is searched as it is: no extra model call, no earlier
  topic mixed in. The rewrite counts as its own step in tokens and cost, obeys local-only, and never switches
  models: on a rate limit or quota it stops and the search uses the previous question plus this one.
- Requests to transform the previous answer also reuse the chunks it cited, re-read from the database through
  the current source map: a deleted, unauthorized or no longer connected source is never reused.
- Broad questions look at 10 candidates and, once one clears the threshold, keep up to 8 chunks within 0.08 of
  it. If nothing clears the threshold the answer is still "I don't know".
- Detailed answers (detail, summary, points, example) may use up to 8 cited sentences (4 in Telugu and Hindi,
  whose sentences cost many more tokens); every sentence is still citation- and claim-checked. Earlier answers
  are context for what is asked, never sources.
- "In Telugu" / "in Hindi" in the question sets the answer language (the chat's language selector still wins).
- A vague question at the start of a new chat ("What are its main features?", "Explain the architecture.") has
  no earlier message to say what "its" is; it scored 0.425 against the threshold of 0.50 and was declined. Such a
  question is now also searched together with the names of the sources this run may search (ready sources in
  the project's scope only, at most five; no model call), which lifted it to 0.730 on the same document. Specific
  first questions are searched exactly as before, and the threshold and checks are unchanged.
