# free-model-aggregator-02 entry

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/dillahhhh/free-model-aggregator-02-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/dillahhhh/free-model-aggregator-02-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

Aggregator server for the [make.fun "The best live free-model aggregator" bounty](https://make.fun/bounty/free-model-aggregator-02) ($500 in USDC).
One server (Node 22, stdlib only) that answers OpenAI- and Anthropic-compatible
requests with real completions from OpenRouter's free models, dialled through
the grader's gateway.

Routing: best-rated free model meeting each prompt's requirements
(JSON -> Nemotron 3.5 Super VL; non-JSON/tools -> Nemotron 3.5 Super VL,
Nemotron 3 Ultra as backup), with failover to the next-best model on refusal,
per-model cooldowns, the `x-makefun-upstream-failures` trail header, and token
validation on every provider answer. Catalog: exactly the 14 dialable free models.

Run: `node main.ts` (registers on a free port, serves `/v1/models`,
`/v1/chat/completions`, and `/v1/messages`).
