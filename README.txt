AGENT PLATFORM EXPLAINED
========================

Audience: a technical person who is new to agents, LangGraph, and context
engineering.

This document explains what we built, why each part exists, and how data flows
through the system. It uses plain language first, then gives technical details.


1. THE SHORT VERSION
====================

We built a local AI agent platform.

It is not just a chatbot.

A normal chatbot mostly does this:

  User asks question
      |
      v
  Model answers from prompt/context

Our research assistant does this:

  User asks research question
      |
      v
  Agent plans research
      |
      v
  Agent searches web
      |
      v
  Agent reads sources
      |
      v
  Agent scores sources
      |
      v
  Agent writes answer with citations
      |
      v
  Agent critic checks quality
      |
      v
  Policy checks look for problems
      |
      v
  Human can approve or request another pass
      |
      v
  Run is saved to history and Neo4j memory

The key idea:

  A chatbot gives an answer.
  An agent runs a workflow.

The platform now has three main end-user workflows:

  1. Research Assistant
     Helps research a question with sources, citations, review, and memory.

  2. Network Design Helper
     Helps a network engineer chat through a design, ground it in company
     standards, create a design package, and prepare a FortiGate handoff.

  3. FortiGate Provisioning Agent
     Creates draft FortiGate design and configuration artifacts for review.
     It does not log in to a firewall or make live changes.


2. WHAT PROBLEM THIS SOLVES
===========================

Large language models are powerful, but by themselves they have weaknesses:

- They can sound confident when they are wrong.
- They can forget what happened in previous sessions.
- They do not automatically know which sources are trustworthy.
- They do not naturally show their work.
- They do not automatically pause for human review.
- They do not automatically create an audit trail.

This platform wraps the model in a system that adds:

- Planning
- Web research
- Source reading
- Citation tracking
- Quality review
- Policy checks
- Human approval
- Memory
- Observability

That turns a model from "answer generator" into a more reliable research
workflow.


3. THE MACHINES INVOLVED
========================

There are two main machines in the current setup.

Machine 1: agent-host.example
----------------------------

This is the agent platform server.

It runs:

- Admin portal
- LangGraph app
- Research Assistant web UI
- LiteLLM
- LiteLLM Postgres database
- Redis
- Neo4j
- Langfuse
- Supporting containers

Important URLs:

  Admin Portal:       http://agent-host.example
  Research UI:        http://agent-host.example:8080
  Agent API:          http://agent-host.example:8001
  LiteLLM UI:         http://agent-host.example:4010/ui
  Neo4j Browser:      http://agent-host.example:7474
  Langfuse:           http://agent-host.example:3001


Machine 2: vllm-host.example
----------------------------

This is the model/inference server.

It runs:

- vLLM

Important URLs:

  vLLM API:           http://vllm-host.example:8000/v1

Important networking detail:

  LiteLLM and the LangGraph API run with host networking on Linux.

This means LiteLLM sees LangGraph requests as coming from the real agent host
LAN address, not from a Docker bridge address like 172.x.x.x. That matters if
you want LiteLLM logs and future access policies to identify the real host path.


High-level machine diagram
--------------------------

  Your Browser
      |
      | http://agent-host.example
      v
  Admin Portal
      |
      | http://agent-host.example:8080
      v
  Research Assistant UI
      |
      | http://agent-host.example:8001
      v
  LangGraph FastAPI App
      |
      | asks model through LiteLLM
      v
  LiteLLM on agent-host.example
      |
      | sends request to vLLM
      v
  vLLM running Gemma 31B


4. MAIN SERVICES
================

Research Assistant UI
---------------------

URL:

  http://localhost:8080

This is the web page users interact with.

It lets a user:

- Ask a research question.
- Choose quick/deep/comparison/decision memo modes.
- See the final research brief.
- See citations and source quality.
- Review policy and quality reports.
- Approve or mark a run as needing work.
- Run a follow-up pass.
- Start an interactive human-review workflow.
- Search Neo4j memory.
- View backlog, audit, and dedup information.


Network Design Helper UI
------------------------

URL:

  http://localhost:8080/network-design

This is a ChatGPT-style page for network design work.

It lets a network engineer:

- Have a normal design conversation.
- Fill in customer, site, and design context in a sidebar.
- Search uploaded standards and reference documents.
- Generate a design package.
- Generate a FortiGate handoff payload.
- Download a design Markdown file.
- Download a raw run JSON file.
- Download a detailed audit Markdown file.
- Generate and download a draft FortiGate .conf file.

The important difference from a normal chatbot:

  The helper does not just answer from memory. It retrieves standards,
  extracts requirements, checks design evidence, and builds an auditable
  package.


FortiGate Agent UI
------------------

URL:

  http://localhost:8080/fortigate

This page is for creating review-ready FortiGate artifacts.

It can:

- Ask intake questions.
- Parse an existing FortiGate config.
- Plan a change.
- Retrieve relevant standards.
- Generate draft CLI configuration.
- Run validation checks.
- Run a model judge.
- Save the package for review.

Important safety rule:

  The FortiGate agent is artifact-only. It does not push changes to a real
  device.


Agent API
---------

URL:

  http://localhost:8001

This is the FastAPI backend.

It owns:

- Chat endpoint
- Research endpoint
- Interactive research endpoint
- Network Design Helper endpoints
- FortiGate agent endpoints
- Standards ingestion and search endpoints
- Memory endpoints
- Graph visualization endpoints


LangGraph
---------

LangGraph is the workflow engine.

Think of LangGraph like a flowchart that can run code.

Each box in the flowchart is called a node.
Each arrow is called an edge.
The data moving through the graph is called state.

Example:

  retrieve_memory -> plan_research -> search_sources -> read_sources

That means:

1. First retrieve memory.
2. Then plan research.
3. Then search for sources.
4. Then read those sources.


LiteLLM
-------

LiteLLM is the model gateway.

The LangGraph app does not talk directly to vLLM. Instead, it calls LiteLLM
using an OpenAI-compatible API.

Current model name:

  gemma-local

Current LiteLLM API URL:

  http://agent-host.example:4010/v1

Why LiteLLM is useful:

- One common API for different models.
- Central place for model routing.
- API key management.
- Usage logging.
- Future ability to switch models without rewriting the agent.


vLLM
----

vLLM runs the actual language model.

In this setup, it serves the Gemma 31B model on the Linux GPU server.

The basic path is:

  LangGraph app -> LiteLLM -> vLLM -> Gemma model


Redis
-----

Redis stores LangGraph checkpoints.

A checkpoint is a saved copy of where the graph is.

This matters because:

- Chat threads can remember prior messages.
- Interactive research can pause and resume.
- Network design runs can preserve workflow state.
- FortiGate workflows can preserve workflow state.
- LangGraph can keep state between steps.


Neo4j
-----

Neo4j stores research memory as a graph.

A graph database stores things as nodes and relationships.

Instead of only storing a research run as one big JSON file, Neo4j lets us ask
relationship questions like:

- Which sources were reused across runs?
- Which claims cited the same source?
- Which research runs were marked needs_work?
- Which run followed up another run?
- Which old run is related to this new question?

Neo4j Browser:

  http://localhost:7474

Login:

  Username: neo4j
  Password: change-me-neo4j-password


Langfuse
--------

Langfuse is observability for LLM applications.

It helps answer:

- What model calls happened?
- How long did they take?
- What did the graph do?
- Did the request reach the model server?
- Where did a failure happen?

URL:

  http://localhost:3001


Standards Library
-----------------

The standards library is where company standards and reference documents go.

The source folder is:

  /data/fortigate-standards/raw

The generated search index is:

  /data/fortigate-standards/index.json

Supported document types include:

- Markdown
- Text
- JSON/YAML
- FortiGate config files
- HTML
- PDF
- Word documents
- PowerPoint decks
- Excel spreadsheets

The standards library is used by both the Network Design Helper and the
FortiGate agent.

The Network Design Helper also extracts standard requirements and builds a
compliance matrix.

Plain English version:

  The system tries to answer, "Which standard requirement caused this design
  decision, and where is the evidence in the generated package?"


Graph Viewers
-------------

The platform includes visual graph pages for the workflows:

  http://localhost:8001/graph
  http://localhost:8001/research/graph
  http://localhost:8001/network-design/graph
  http://localhost:8001/fortigate/graph

These pages render Mermaid graphs in the browser and include zoom, pan, and
download controls.


5. WHAT IS AN AGENT?
====================

A simple LLM call looks like this:

  Prompt -> Model -> Answer

An agent is more like this:

  Goal
    -> decide what to do
    -> use tools
    -> inspect results
    -> revise plan
    -> produce final answer

In this project, the agent does not have unlimited freedom. It follows a
designed workflow.

That is important.

We are not saying:

  "Model, do anything you want."

We are saying:

  "Model, perform this research workflow step by step."

This is safer, easier to debug, and easier to improve.


6. WHAT IS CONTEXT ENGINEERING?
===============================

Prompt engineering is about writing a good instruction.

Context engineering is bigger.

Context engineering means deciding what information the model gets, when it gets
it, and in what format.

Examples from this project:

- The planning node gets the user question and memory from Neo4j.
- The synthesis node gets source text, source scores, and constraints.
- The critic node gets the answer, citations, evidence, and limitations.
- The policy node gets structured quality data.
- The human review node gets the draft answer and policy report.

The model is not just given one giant blob of text.

Each node gets the context it needs for that job.


7. RESEARCH WORKFLOW IN DETAIL
==============================

The current research graph looks like this:

  retrieve_memory
      |
      v
  plan_research
      |
      v
  search_sources
      |
      v
  read_sources
      |
      v
  score_sources
      |
      v
  synthesize_research
      |
      v
  analyze_gaps
      |
      +--------------------+
      |                    |
      v                    v
  deepening pass       critique_research
      |                    |
      v                    v
  resynthesize        revise_final_answer
      |                    |
      +---------+----------+
                |
                v
          check_policy
                |
      +---------+----------+
      |                    |
      v                    v
  policy repair        verify_citations
      |                    |
      v                    v
  resynthesize             END


Each node explained
-------------------

retrieve_memory

  Looks in Neo4j before doing new research.

  It asks:

  - Have we researched this before?
  - Are there related runs?
  - Are there prior claims?
  - Are there repeated/trusted sources?

  This makes the assistant less repetitive.


plan_research

  Creates a research plan.

  It decides:

  - What should be checked?
  - What kinds of sources are needed?
  - What is likely uncertain?


search_sources

  Searches the web.

  Primary search:

    Tavily

  Fallback search:

    DuckDuckGo HTML


read_sources

  Fetches web pages and extracts readable text.


score_sources

  Scores sources for:

  - Authority
  - Relevance
  - Risk
  - Overall score


synthesize_research

  Writes the first research answer using the evidence.

  It returns:

  - Answer
  - Findings
  - Citations
  - Confidence
  - Follow-up questions


analyze_gaps

  Looks at the first answer and asks:

  - What is missing?
  - What needs more depth?
  - What is unclear?

  In deep/comparison modes, this can trigger another search pass.


critique_research

  Acts like a skeptical reviewer.

  It looks for:

  - Unsupported claims
  - Weak citations
  - Missing perspectives
  - Source risks


revise_final_answer

  Rewrites the answer using the critic's feedback.


check_policy

  Runs deterministic checks.

  Examples:

  - Are there enough citations?
  - Are there unsupported claims?
  - Is source diversity too low?
  - Did deep mode actually do a deeper pass?


policy repair

  If the policy report finds problems, the graph can do one targeted repair pass.

  Example:

  If citations are weak, it searches for stronger sources.


verify_citations

  Checks whether the cited evidence actually supports the claims.


8. HUMAN IN THE LOOP
====================

There are two human-review patterns.


Pattern 1: Review after a saved run
-----------------------------------

The graph completes.

Then the user can:

- Approve final answer.
- Mark needs work.
- Add notes.
- Select issues.
- Run follow-up from review.

This is practical and easy to understand.


Pattern 2: True LangGraph pause/resume
--------------------------------------

This is more advanced.

The graph pauses before it is fully done.

The node is:

  human_review_checkpoint

The graph uses LangGraph interrupt/resume:

  interrupt()
  Command(resume=...)

Flow:

  graph runs
      |
      v
  reaches human_review_checkpoint
      |
      v
  pauses and waits for user
      |
      v
  user approves or requests repair
      |
      v
  graph resumes from the checkpoint

Why this matters:

- The system does not need to start over.
- The human decision becomes part of the workflow.
- The review is auditable.
- This pattern can later support approvals, compliance, and multi-user review.


9. MEMORY IN NEO4J
==================

The research assistant saves memory into Neo4j after runs complete.

The graph shape is:

  ResearchRun
      |
      +-- ANSWERED --> Question
      |
      +-- USED_SOURCE --> Source
      |
      +-- MADE_CLAIM --> Claim
      |
      +-- HAS_REVIEW --> HumanReview
      |
      +-- HAS_POLICY --> PolicyReport
      |
      +-- HAS_QUALITY --> QualityReport
      |
      +-- FOLLOWED_BY --> ResearchRun

And:

  Claim -- CITED --> Source


Why this is useful
------------------

A normal database can store rows.

Neo4j stores relationships.

That means we can ask questions like:

  "Which runs used this same source?"

  "Which claims cite this source?"

  "Which research runs still need work?"

  "Which follow-up fixed an earlier problem?"

  "Have we already researched this question?"


Current memory features
-----------------------

Retrieval before web search

  Before researching, the graph asks Neo4j for related prior runs, prior claims,
  and reusable sources.


Research backlog

  Finds runs with:

  - Weak citations
  - Unsupported claims
  - Missing perspectives
  - Policy warnings
  - Human needs_work review


Audit

  Shows the review, policy, quality, parent run, and follow-up runs for a
  selected research run.


Dedup

  Finds repeated sources and repeated claims across runs.


10. STORAGE LAYERS
==================

This system has several kinds of storage.


Redis
-----

Purpose:

  Short-term graph checkpoint state.

Used for:

- Chat memory by thread_id
- LangGraph state
- Interactive pause/resume


JSON files
----------

Purpose:

  Simple durable archive of completed research runs.

Location inside container:

  /data/research-runs

Why keep it:

- Easy to inspect
- Easy to back up
- Fallback if Neo4j is down


Neo4j
-----

Purpose:

  Queryable research memory graph.

Used for:

- Source reuse
- Claim reuse
- Backlog
- Audits
- Dedup
- Retrieval before research


Langfuse databases
------------------

Purpose:

  Observability and traces.

Used for:

- LLM call traces
- Timing
- Debugging
- Request visibility


11. FRONTEND GUIDE
==================

Research Assistant:

  http://localhost:8080

Network Design Helper:

  http://localhost:8080/network-design

FortiGate Agent:

  http://localhost:8080/fortigate


Research Assistant main controls
--------------------------------

Research question

  The question the user wants answered.


Mode

  quick:
    Faster, smaller research pass.

  deep:
    More likely to do iterative deepening.

  comparison:
    Better for comparing tools, products, or options.

  decision_memo:
    Better for recommendation-style answers.


Constraints/context

  Optional instructions such as:

  - Prefer official docs.
  - Focus on local stack.
  - Explain uncertainty.
  - Compare cost and operational complexity.


Run structured research

  Runs the normal graph all the way through.


Run Interactive Review

  Runs the graph until the human review checkpoint, then pauses for approval or
  repair.


Research Brief
--------------

This is the main output.

It includes:

- Answer
- Policy report
- Quality report
- Retrieved research memory
- Findings
- Detailed synthesis
- Deepened areas
- Citations and verification
- Sources and source quality
- Execution trace
- Human review


Memory panel
------------

The Memory panel lets you inspect Neo4j.

Buttons:

  Search Memory
    Search prior research runs.

  Current Run Memory
    Show graph memory for the loaded run.

  Backlog
    Show unresolved research work.

  Dedup
    Show repeated claims and sources.

  Audit Run
    Show review/policy/follow-up context for the selected run.


Network Design Helper controls
------------------------------

The Network Design Helper is organized around a main chat panel and a sidebar.

Chat panel:

  This is where the engineer talks through the design.

Sidebar:

  This is where the engineer enters structured context such as customer, site,
  design goal, constraints, FortiGate platform, WAN details, VLANs, routing,
  logging, and security requirements.

Important buttons:

  Search Standards
    Looks through the uploaded standards index.

  Generate Design Package
    Runs the Network Design Helper graph and creates the package.

  Generate FortiGate Config
    Sends the handoff to the FortiGate agent to produce draft CLI.

  Design .md
    Downloads the design package as Markdown.

  Config .conf
    Downloads the generated FortiGate draft CLI configuration.

  Audit .md
    Downloads an audit file with standards evidence, extracted requirements,
    compliance matrix, traces, and judge/config details when available.

  Run .json
    Downloads the raw run response.


FortiGate Agent controls
------------------------

The FortiGate page collects FortiGate-specific intake, can accept existing
configuration text, and generates draft artifacts for review.

It is useful for:

- New FortiGate site design.
- Change planning against an existing config.
- SD-WAN design.
- Firewall policy intent.
- Standards-grounded config generation.
- Review packages before implementation.


12. API GUIDE
=============

Base URL:

  http://localhost:8001


Health
------

  GET /health

Example:

  curl http://localhost:8001/health


Chat
----

  POST /chat

Example:

  curl -sS http://localhost:8001/chat \
    -H 'Content-Type: application/json' \
    -d '{"thread_id":"demo","message":"Say hello from LangGraph."}'


Research
--------

  POST /research

Runs the full research graph to completion.

Example:

  curl -sS http://localhost:8001/research \
    -H 'Content-Type: application/json' \
    -d '{"thread_id":"demo","mode":"quick","question":"What is LangGraph interrupt/resume useful for?","constraints":"Prefer official docs."}'


Saved runs
----------

  GET /research/runs
  GET /research/runs/{run_id}


Human review
------------

  POST /research/runs/{run_id}/review

Example body:

  {
    "decision": "approved",
    "reviewer_notes": "Looks good.",
    "selected_issues": []
  }


Follow-up pass
--------------

  POST /research/runs/{run_id}/follow-up

Example body:

  {
    "mode": "quick",
    "instruction": "Run one more targeted pass on weak citations."
  }


Interactive research
--------------------

Start:

  POST /research/interactive

Resume:

  POST /research/interactive/{thread_id}/review


Network Design Helper
---------------------

  POST /network-design/message
  POST /network-design/chat
  GET  /network-design/runs
  GET  /network-design/runs/{run_id}
  GET  /network-design/standards/search?q=...

Example:

  curl -sS http://localhost:8001/network-design/chat \
    -H 'Content-Type: application/json' \
    -d '{"thread_id":"network-demo","intake":{"customer_name":"ExampleCo","site_name":"branch-001","design_goal":"Create a dual-WAN branch design with guest internet and FortiAnalyzer logging."},"messages":[{"role":"user","content":"Use SD-WAN failover, separate corp and guest zones, and prepare a FortiGate handoff."}]}'

Important response fields:

  standard_requirements
    Structured requirements extracted from retrieved standards.

  compliance_matrix
    Maps requirements to evidence in the design and handoff.

  fortigate_handoff
    Structured payload that can be sent to the FortiGate agent.

  markdown
    Downloadable design package text.


FortiGate Agent
---------------

  POST /fortigate/design
  POST /fortigate/interactive
  POST /fortigate/interactive/{thread_id}/resume
  POST /fortigate/configs/parse
  POST /fortigate/changes/analyze
  POST /fortigate/runs/{run_id}/judge
  POST /fortigate/runs/{run_id}/review
  POST /fortigate/standards/ingest
  GET  /fortigate/standards/search?q=...
  GET  /fortigate/runs
  GET  /fortigate/runs/{run_id}


Graph viewers
-------------

  GET /graph
  GET /research/graph
  GET /network-design/graph
  GET /fortigate/graph

Raw Mermaid source:

  GET /graph/mermaid
  GET /research/graph/mermaid
  GET /network-design/graph/mermaid
  GET /fortigate/graph/mermaid


Memory
------

  GET /memory/health
  GET /memory/research/search?q=...
  GET /memory/research/runs/{run_id}
  GET /memory/research/runs/{run_id}/audit
  GET /memory/research/sources/{source_hash}/runs
  GET /memory/research/backlog
  GET /memory/research/dedup?q=...


13. EXAMPLE END-TO-END FLOW
===========================

Imagine the user asks:

  "Compare NVIDIA RTX GPUs for local LLM inference."

The system does:

Step 1: Retrieve memory

  Neo4j checks whether similar GPU research has already happened.


Step 2: Plan

  The planner decides it needs things like:

  - GPU memory sizes
  - Inference speed
  - Power usage
  - Price
  - Model fit
  - Official specs


Step 3: Search

  The agent searches for relevant sources.


Step 4: Read

  It fetches source pages.


Step 5: Score

  It decides which sources are stronger or weaker.


Step 6: Synthesize

  It writes a draft answer with citations.


Step 7: Critique

  It checks for weak evidence or missing perspectives.


Step 8: Policy check

  It checks citation count, diversity, confidence, and weak claims.


Step 9: Human review

  The user can approve or ask for one more pass.


Step 10: Save memory

  The completed run is saved to JSON and Neo4j.


14. WHY THIS IS BETTER THAN GENERAL CHAT
========================================

General chat:

  - Often gives an answer immediately.
  - May not show source quality.
  - May not remember prior research.
  - May not create audit trails.
  - May not pause for review.

This research assistant:

  - Plans first.
  - Searches and reads sources.
  - Scores sources.
  - Produces structured citations.
  - Verifies citations.
  - Critiques itself.
  - Applies policy checks.
  - Lets a human approve or repair.
  - Saves research memory.
  - Retrieves prior work before new research.


15. CURRENT LIMITATIONS
=======================

This is still a local development system.

Known limitations:

- OPA is not wired yet.
- User accounts and roles are not implemented.
- Human approvals do not yet record a named user.
- Search quality depends on Tavily or DuckDuckGo availability.
- Some pages may not fetch cleanly.
- Neo4j writes are best-effort.
- JSON is still the fallback archive.
- Source reputation is basic and not yet learned over long periods.
- No production auth layer is in front of the API.


16. GOOD NEXT STEPS
===================

High-value future improvements:

1. Backfill all old JSON runs into Neo4j.

2. Add source reputation memory:

   If a source is repeatedly weak, down-rank it.
   If official docs are repeatedly strong, prefer them.

3. Add stale fact detection:

   If a claim is old, force fresh verification.

4. Add user identity:

   Record who approved a research brief.

5. Add OPA:

   Move policy checks into an external policy engine.

6. Add richer Neo4j visualizations:

   Show research run -> claim -> source paths in the frontend.

7. Add scheduled backlog review:

   Surface unresolved research items automatically.


17. QUICK GLOSSARY
==================

Agent

  A system that uses a model as part of a workflow to do a task.


Node

  One step in a LangGraph workflow.


Edge

  A connection from one node to another.


State

  The data passed through the graph.


Checkpoint

  A saved graph state that can be resumed later.


Interrupt

  A LangGraph feature that pauses a running graph and waits for external input.


Resume

  Continuing a paused graph using new input.


Context engineering

  Choosing the right information to give the model at the right step.


Citation

  A source-backed claim in the answer.


Policy check

  A deterministic rule-based check for quality or safety requirements.


Critic

  A model step that reviews the answer for weaknesses.


Neo4j

  A graph database used to store relationships between runs, claims, sources,
  reviews, and follow-ups.


Langfuse

  Observability for LLM calls and traces.


LiteLLM

  A model gateway that provides an OpenAI-compatible API in front of vLLM.


vLLM

  The inference server that runs the actual model.


18. SIMPLE MENTAL MODEL
=======================

If you remember only one thing, remember this:

  The model is the engine.
  LangGraph is the workflow.
  Redis is the checkpoint memory.
  Neo4j is the research memory.
  Langfuse is the flight recorder.
  LiteLLM is the model gateway.
  The frontend is the cockpit.

Together, they make the research assistant more than a chat box.

