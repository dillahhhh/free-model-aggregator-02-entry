# free-model-aggregator-02 entry

Aggregator server for the make.fun free-model-aggregator-02 competition.
One server (Node 22, stdlib only) that answers OpenAI- and Anthropic-compatible
requests with real completions from OpenRouter's free models, dialled through
the grader's gateway.

Routing: best-rated free model meeting each prompt's requirements
(JSON -> Nemotron 3.5 Super VL; non-JSON/tools -> Inkling 2 Small), with
failover to the next-best model on refusal, per-model cooldowns, the
`x-makefun-upstream-failures` trail header, and token validation on every
provider answer. Catalog: exactly the 14 dialable free models.
