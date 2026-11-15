// free-model-aggregator-02 — candidate v1: quality-first routing with probe-driven
// preferences, failover with a trail header, per-model cooldown, and token validation.
//
// Design (from the audit + probe plan):
//   - Routing picks, for each prompt, the best model that meets its requirements.
//     The order comes from ROUTING_PREFERENCE below (probe data), with the brief's
//     rating_milli as the fallback order for anything the table does not name.
//     Rating is NOT quality: a model only earns its rating when it actually answers
//     and its body carries the prompt's token, so the preference table is the one
//     place routing changes when probe results arrive — the failover logic below
//     never hardcodes a model.
//   - Failover: a refused call (429/5xx/unreachable) is retried ONCE on the same
//     model when the refusal is transient, then the prompt moves to the next
//     candidate. Every refusal is recorded in the x-makefun-upstream-failures
//     header (at most 8 rows): that header is the only way the failover column
//     can see a refusal that was retried and then served.
//   - Cooldown: a model that refused is never dialled again in this trial
//     (unless it is the only candidate left for a prompt). Refusals are not
//     counted in upstream_calls, so a healthy failover costs no tie-break calls.
//   - Token validation: before any answer is returned, the provider's own body
//     must carry the prompt's token. An answer without the token earns 0 quality,
//     so it is treated as a failure and the prompt fails over to the next model.
//     A successful-but-tokenless call still counts as an answered upstream call.
//   - Call economy: one answered call per prompt when the first call succeeds.
//     No speculative or probe calls inside a trial.
//   - Determinism: every choice is a function of the brief, the requests and
//     the gateway's answers so far. No wall clock, no unseeded randomness; the
//     env is polled one request at a time, in arrival order.
//
// What a trial gives this server (same contract as the starter):
//   - MAKEFUN_ENV_URL: POST /register {protocol_version: 1, port, catalog},
//     then GET /work until {"done": true}, then exit 0.
//   - MAKEFUN_PROMPT_FILE: the brief — the 14 free models (id, rating_milli,
//     tools/vision/json/context_length) and the 6 prompts (id, api, stream,
//     weight, requirements), in the order the env sends them.
//   - OPENAI_BASE_URL / OPENAI_API_KEY: the grader's gateway (Chat Completions
//     only) and the trial's dummy key, which the gateway swaps for the entrant's
//     own pushed OpenRouter key.
//   - `node main.ts --list` prints the catalog document and exits 0; the
//     register body's catalog is that same document, byte for byte.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The trail header's row cap: the env reads at most 8 rows, and more voids it. */
const MAX_TRAIL_ROWS = 8;
/** The trail header's byte cap: the env reads nothing over 8 KiB. */
const MAX_TRAIL_BYTES = 8 * 1024;
/** The most bytes of one refused gateway body this server reads for its error code. */
const MAX_ERROR_BYTES = 4096;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;
/** Models that refused a call in this trial: never dialled again unless nothing else is left. */
const cooledDown = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator-v1: ${message}\n`);
}

/**
 * Probe-derived routing preference, by requirement class. These are the models
 * the probe plan expects to answer best, in order; anything the table does not
 * name falls back to the brief's rating order. UPDATE THIS TABLE — and nothing
 * else — when probe results arrive: if a model 429s, answers slowly, or answers
 * without the token, move it down or drop it from its class. A model that is
 * not in the brief is skipped silently, so the table stays valid if a final
 * instance lists different models or ratings.
 *
 * Initial values (probe plan, 15 Nov): inkling first for every non-JSON
 * prompt (rating 1475), super-vl first for JSON (rating 1303, best JSON-capable),
 * ultra and gemma-4-31b as the fallbacks the probes will confirm or reorder.
 */
const ROUTING_PREFERENCE: Readonly<Record<string, readonly string[]>> = {
  json: ['openrouter/nvidia/nemotron-3.5-super-vl:free', 'openrouter/google/gemma-4-31b-it:free'],
  tools: ['openrouter/thinkingmachines/inkling-2-small:free', 'openrouter/nvidia/nemotron-3.5-super-vl:free', 'openrouter/nvidia/nemotron-3-ultra-550b-a55b:free'],
  plain: ['openrouter/thinkingmachines/inkling-2-small:free', 'openrouter/nvidia/nemotron-3.5-super-vl:free', 'openrouter/nvidia/nemotron-3-ultra-550b-a55b:free'],
};

/** The requirement class a prompt belongs to, for the preference table. */
function preferenceKey(requires: Requirement): string {
  if (requires.json === true) return 'json';
  if (requires.tools === true) return 'tools';
  return 'plain';
}

/**
 * The candidates for one prompt, best first: every free model that meets its
 * requirements, ordered by the probe preference table first (in listed order),
 * then by the brief's rating_milli (ties broken by id, so the order is stable
 * across runs). A model the table names but the brief does not list never appears.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const eligible = freeModels(brief).filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  eligible.sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
  const preferred = ROUTING_PREFERENCE[preferenceKey(requires)] ?? [];
  const rank = new Map(preferred.map((id, i) => [id, i]));
  const ordered = [...eligible];
  ordered.sort((a, b) => {
    const ra = rank.has(a.id) ? (rank.get(a.id) as number) : preferred.length;
    const rb = rank.has(b.id) ? (rank.get(b.id) as number) : preferred.length;
    return ra - rb;
  });
  return ordered;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** A prompt token looks like SPARROW-7435: an uppercase word, a dash, digits. */
const TOKEN_PATTERN = /\b[A-Z]{2,}-[0-9]+\b/;

/** The unique token the arriving request carries, or null when it carries none. */
function tokenOfRequest(body: Json): string | null {
  const texts: string[] = [];
  const system = body.system;
  if (typeof system === 'string') texts.push(system);
  else if (Array.isArray(system)) {
    for (const block of system) if (isRecord(block) && typeof block.text === 'string') texts.push(block.text);
  }
  if (Array.isArray(body.messages)) {
    for (const raw of body.messages) {
      if (!isRecord(raw)) continue;
      const content = raw.content;
      if (typeof content === 'string') texts.push(content);
      else if (Array.isArray(content)) {
        for (const block of content) if (isRecord(block) && typeof block.text === 'string') texts.push(block.text);
      }
    }
  }
  for (const text of texts) {
    const found = TOKEN_PATTERN.exec(text);
    if (found !== null) return found[0];
  }
  return null;
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One refusal, as the trail header reports it: the gateway's status, its error code, the model. */
interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

/** The trail header for a response, or nothing when no call was refused. */
function trailHeaders(trail: readonly TrailRow[]): Record<string, string> {
  if (trail.length === 0) return {};
  const value = JSON.stringify(trail.slice(0, MAX_TRAIL_ROWS));
  if (Buffer.byteLength(value) > MAX_TRAIL_BYTES) return {};
  return { 'x-makefun-upstream-failures': value };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, trail: readonly TrailRow[] = []): void {
  const headers = trailHeaders(trail);
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message), headers);
  else sendJson(res, status, openAIError(message, type, null), headers);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** A gateway error body's code, or a default for its status when it carries none. */
function errorCodeOfText(text: string, status: number): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      if (typeof value.error.code === 'string' && value.error.code !== '') return value.error.code.slice(0, 64);
      if (typeof value.error.type === 'string' && value.error.type !== '') return value.error.type.slice(0, 64);
    }
  } catch {
    // Not JSON: fall through to the status default.
  }
  if (status === 429) return 'rate_limit';
  if (status === 401) return 'unauthorized';
  if (status >= 500) return 'upstream_error';
  return 'upstream_refused';
}

/** What one gateway call returned: an answer (JSON or buffered stream) or a refusal. */
type DialOutcome =
  | { kind: 'answered'; json: Json | null; frames: string[] | null; text: string }
  | { kind: 'refused'; status: number; code: string };

/** The text one OpenAI SSE frame carries: its choices' delta content, if any. */
function openAIStreamText(frame: string): string {
  let out = '';
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') continue;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (!isRecord(chunk)) continue;
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : null;
      if (delta !== null && typeof delta.content === 'string') out += delta.content;
    } catch {
      // Not JSON: nothing to carry.
    }
  }
  return out;
}

/** The text one Anthropic SSE event carries: its text_delta, if it is one. */
function anthropicEventText(event: string): string {
  for (const line of event.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    try {
      const data: unknown = JSON.parse(payload);
      if (!isRecord(data)) continue;
      if (data.type === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
        return data.delta.text;
      }
    } catch {
      // Not JSON: nothing to carry.
    }
  }
  return '';
}

/**
 * One upstream stream, buffered whole: the relay frames (every `model` the id
 * dialled) and the answer's assembled text. Buffering is what makes token
 * validation possible on a stream: nothing is sent to the client until the
 * provider's text is known to carry the token, so a bad stream never reaches
 * the client half-sent. The frames keep their order, so the wire checks see
 * the same event sequence a live relay would send.
 */
async function collectStream(body: ReadableStream<Uint8Array>, wire: 'openai' | 'anthropic', model: string): Promise<{ frames: string[]; text: string }> {
  const frames: string[] = [];
  let text = '';
  if (wire === 'openai') {
    for await (const frame of openAIStreamWithModel(body, model)) {
      frames.push(frame);
      text += openAIStreamText(frame);
    }
  } else {
    for await (const event of anthropicEventsFromOpenAI(body, model)) {
      frames.push(event);
      text += anthropicEventText(event);
    }
  }
  return { frames, text };
}

/**
 * One dial of one candidate for one prompt. A refusal carries its status and
 * code; an answer carries its body (JSON) or its buffered stream frames, plus
 * the assembled provider text the token check reads.
 */
async function dialCandidate(candidate: ModelRow, clientBody: Json, wire: 'openai' | 'anthropic', stream: boolean): Promise<DialOutcome> {
  const upstreamBody = wire === 'openai' ? { ...clientBody, model: candidate.id } : anthropicToOpenAI(clientBody, candidate.id);
  let upstream: Response;
  try {
    upstream = await dial(candidate.id, upstreamBody);
  } catch {
    return { kind: 'refused', status: 502, code: 'upstream_unreachable' };
  }
  if (!upstream.ok || upstream.body === null) {
    const text = await upstream.text().catch(() => '');
    return { kind: 'refused', status: upstream.status, code: errorCodeOfText(text.slice(0, MAX_ERROR_BYTES), upstream.status) };
  }
  try {
    if (stream) {
      const collected = await collectStream(upstream.body, wire, candidate.id);
      return { kind: 'answered', json: null, frames: collected.frames, text: collected.text };
    }
    const answer: unknown = await upstream.json();
    const json = isRecord(answer) ? answer : {};
    return { kind: 'answered', json, frames: null, text: JSON.stringify(json) };
  } catch {
    return { kind: 'refused', status: 502, code: 'bad_upstream_body' };
  }
}

/** What serving one prompt produced: a validated answer, or the prompt lost. */
type AttemptResult =
  | { ok: true; model: string; trail: TrailRow[]; json: Json | null; frames: string[] | null }
  | { ok: false; message: string; trail: TrailRow[] };

/** Whether a refusal is worth one retry on the same model: a 429 or a 5xx (and a lost connection). */
function isTransient(status: number): boolean {
  return status === 429 || status >= 500;
}

/**
 * Serve one prompt: walk the candidates best-first. A refused call is
 * retried once on the same model when transient, then the model cools down
 * for the rest of the trial and the prompt moves to the next candidate. An
 * answer whose provider body does not carry the prompt's token earns nothing,
 * so it fails over too (the answered call still counts upstream — no trail
 * row, because it was not a refusal). The first validated answer wins, with
 * the trail of every refusal behind it.
 */
async function attemptPrompt(clientBody: Json, wire: 'openai' | 'anthropic', requires: Requirement, token: string | null, stream: boolean): Promise<AttemptResult> {
  const trail: TrailRow[] = [];
  let candidates = candidatesFor(requires);
  const usable = candidates.filter((model) => !cooledDown.has(model.id));
  if (usable.length > 0) candidates = usable;
  for (const candidate of candidates) {
    let retried = false;
    for (;;) {
      const dialled = await dialCandidate(candidate, clientBody, wire, stream);
      if (dialled.kind === 'answered') {
        if (token !== null && !dialled.text.includes(token)) {
          log(`token missing from ${candidate.id} answer; failing over`);
          break;
        }
        return { ok: true, model: candidate.id, trail, json: dialled.json, frames: dialled.frames };
      }
      trail.push({ status: dialled.status, code: dialled.code, model: candidate.id });
      if (isTransient(dialled.status) && !retried) {
        retried = true;
        continue;
      }
      cooledDown.add(candidate.id);
      break;
    }
  }
  return { ok: false, message: 'every free model that meets this prompt refused it or answered without its token', trail };
}

/**
 * One client completion request, on either wire: route it, dial it, validate
 * the token, and answer it — or lose the prompt with a 502 that still carries
 * the trail of the refusals behind it.
 */
async function serveOne(req: IncomingMessage, res: ServerResponse, wire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  const token = tokenOfRequest(body);
  const stream = body.stream === true;
  const result = await attemptPrompt(body, wire, requires, token, stream);
  if (!result.ok) {
    log(`prompt lost: ${result.message}`);
    return sendError(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', result.message, result.trail);
  }
  const headers = trailHeaders(result.trail);
  if (result.trail.length > 0) log(`served by ${result.model} after ${result.trail.length} refusal(s)`);
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for (const frame of result.frames ?? []) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  if (wire === 'openai') sendJson(res, 200, openAIJsonWithModel(result.json ?? {}, result.model), headers);
  else sendJson(res, 200, openAIToAnthropic(result.json ?? {}, result.model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serveOne(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serveOne(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
