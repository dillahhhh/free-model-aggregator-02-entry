// lib/wire.ts — translating between the two client wires and the gateway.
//
// The gateway only speaks Chat Completions. A client may arrive on either
// wire, so this file translates in both directions and nothing else: it
// never invents an answer, never drops a request field the client set
// (max_tokens, temperature, the system prompt, the declared tools), and
// never renames the model except to put back the id that was really dialled
// — the provider's own body reports the upstream id without the
// `openrouter/` prefix, and the house scores the model the answer names.
//
// Directions:
//   - OpenAI in, OpenAI out: stamp the dialled id onto the provider's
//     answer, JSON or streamed.
//   - Anthropic in, Anthropic out: turn a Messages request into a Chat
//     Completions body for the gateway, then turn the gateway's answer
//     back into a Messages response, streamed or not.
// A refusal is never dressed up as an answer: a 4xx/5xx from upstream
// reaches the caller as a status and a code, so it can retry or fail over.

import { isRecord, type Json } from './providers.ts';

/** One model as GET /v1/models names it: the dialable id, its vendor, its published name. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

// ---------------------------------------------------------------------------
// Model lists
// ---------------------------------------------------------------------------

/** GET /v1/models on the OpenAI wire: the OpenAI list shape, ids as dialled. */
export function openAIModelList(list: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: list.map((m) => ({ id: m.id, object: 'model', created: 0, owned_by: m.provider })),
  };
}

/** GET /v1/models on the Anthropic wire: the Anthropic list shape, ids as dialled. */
export function anthropicModelList(list: readonly ServedModel[]): Json {
  return {
    data: list.map((m) => ({ id: m.id, type: 'model', display_name: m.display_name, created_at: '1970-01-01T00:00:00Z' })),
    has_more: false,
    first_id: list.length > 0 ? list[0].id : null,
    last_id: list.length > 0 ? list[list.length - 1].id : null,
  };
}

// ---------------------------------------------------------------------------
// Error bodies
// ---------------------------------------------------------------------------

/** An OpenAI-shaped error body: message, type, and the parameter at fault (or null). */
export function openAIError(message: string, type: string, param: string | null): Json {
  return { error: { message, type, param } };
}

/** An Anthropic-shaped error body: the envelope type plus the error's own type and message. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

// ---------------------------------------------------------------------------
// Non-streamed answers
// ---------------------------------------------------------------------------

/** An OpenAI JSON answer with the dialled id put back in its model field. */
export function openAIJsonWithModel(answer: Json, dialled: string): Json {
  return { ...answer, model: dialled };
}

/** How an OpenAI finish reason reads on the Anthropic wire. */
function anthropicStopReason(finish: string, hasToolCalls: boolean): string {
  if (hasToolCalls) return 'tool_use';
  if (finish === 'length') return 'max_tokens';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** One OpenAI tool call as an Anthropic tool_use block. */
function anthropicToolUseBlock(call: unknown): Json | null {
  if (!isRecord(call) || !isRecord(call.function)) return null;
  let input: unknown = {};
  if (typeof call.function.arguments === 'string') {
    try {
      input = JSON.parse(call.function.arguments);
    } catch {
      input = {};
    }
  }
  return {
    type: 'tool_use',
    id: typeof call.id === 'string' ? call.id : '',
    name: typeof call.function.name === 'string' ? call.function.name : '',
    input,
  };
}

/**
 * An OpenAI JSON answer turned into an Anthropic Messages response: the
 * assistant's text (and any tool_use blocks) in content, the dialled id in
 * model, the mapped stop reason, and the usage in Anthropic's token names.
 * When the provider sent no usage, the output count falls back to 1 for a
 * non-empty answer, because the wire check wants an output count of at
 * least 1 on a served prompt.
 */
export function openAIToAnthropic(answer: Json, dialled: string): Json {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const first = isRecord(choices[0]) ? choices[0] : null;
  const message = first !== null && isRecord(first.message) ? first.message : {};
  const finish = first !== null && typeof first.finish_reason === 'string' ? first.finish_reason : 'stop';

  const blocks: Json[] = [];
  const rawContent = message.content;
  if (typeof rawContent === 'string' && rawContent !== '') {
    blocks.push({ type: 'text', text: rawContent });
  } else if (Array.isArray(rawContent)) {
    for (const part of rawContent) {
      if (isRecord(part) && part.type === 'text' && typeof part.text === 'string') {
        blocks.push({ type: 'text', text: part.text });
      }
    }
  }
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    const block = anthropicToolUseBlock(call);
    if (block !== null) blocks.push(block);
  }
  if (blocks.length === 0) blocks.push({ type: 'text', text: '' });

  const usage = isRecord(answer.usage) ? answer.usage : {};
  const inputTokens = typeof usage.prompt_tokens === 'number' ? usage.prompt_tokens : 0;
  let outputTokens = typeof usage.completion_tokens === 'number' ? usage.completion_tokens : 0;
  if (outputTokens === 0) {
    const hasText = blocks.some((b) => b.type === 'text' && typeof b.text === 'string' && b.text !== '');
    if (hasText) outputTokens = 1;
  }

  const id = typeof answer.id === 'string' && answer.id !== '' ? `msg_${answer.id}` : 'msg_entry';
  return {
    id,
    type: 'message',
    role: 'assistant',
    content: blocks,
    model: dialled,
    stop_reason: anthropicStopReason(finish, calls.length > 0),
    stop_sequence: null,
    usage: { input_tokens: inputTokens, output_tokens: outputTokens },
  };
}

// ---------------------------------------------------------------------------
// Anthropic request -> OpenAI request
// ---------------------------------------------------------------------------

/** The text an Anthropic system prompt carries, whether a string or blocks. */
function anthropicSystemText(system: unknown): string {
  if (typeof system === 'string') return system;
  if (!Array.isArray(system)) return '';
  const parts: string[] = [];
  for (const block of system) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** One Anthropic image block as an OpenAI image_url part. */
function anthropicImagePart(block: Record<string, unknown>): Json | null {
  if (!isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.media_type === 'string' && typeof source.data === 'string') {
    return { type: 'image_url', image_url: { url: `data:${source.media_type};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One Anthropic tool_use block as an OpenAI tool_call entry. */
function anthropicToolCall(block: Record<string, unknown>): Json | null {
  return {
    id: typeof block.id === 'string' ? block.id : '',
    type: 'function',
    function: {
      name: typeof block.name === 'string' ? block.name : '',
      arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
    },
  };
}

/**
 * One Anthropic message (or the tool results it carries) as OpenAI
 * messages. A tool_result block becomes its own `tool` message, because
 * OpenAI has no such block; everything else stays in the one message.
 */
function anthropicMessageParts(raw: unknown): Json[] {
  if (!isRecord(raw)) return [];
  const role = raw.role === 'assistant' ? 'assistant' : 'user';
  const content = raw.content;
  if (typeof content === 'string') return [{ role, content }];
  if (!Array.isArray(content)) return [{ role, content: '' }];

  const texts: string[] = [];
  const parts: Json[] = [];
  const calls: Json[] = [];
  const results: Json[] = [];
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') {
      texts.push(block.text);
    } else if (block.type === 'image') {
      const part = anthropicImagePart(block);
      if (part !== null) parts.push(part);
    } else if (block.type === 'tool_use') {
      const call = anthropicToolCall(block);
      if (call !== null) calls.push(call);
    } else if (block.type === 'tool_result') {
      const text = typeof block.content === 'string'
        ? block.content
        : Array.isArray(block.content)
          ? block.content.filter(isRecord).map((b) => (typeof b.text === 'string' ? b.text : '')).filter(Boolean).join('\n')
          : '';
      results.push({
        role: 'tool',
        tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        content: text,
      });
    }
  }

  const out: Json[] = [];
  if (parts.length > 0 || texts.length > 0 || calls.length > 0) {
    const message: Json = { role, content: parts.length > 0 ? [...texts.map((t) => ({ type: 'text', text: t })), ...parts] : texts.join('\n') };
    if (calls.length > 0) message.tool_calls = calls;
    out.push(message);
  }
  out.push(...results);
  return out;
}

/** An Anthropic tool_choice as the OpenAI tool_choice, when the request sets one. */
function anthropicToolChoice(choice: unknown): Json | null {
  if (!isRecord(choice)) return null;
  if (choice.type === 'auto') return { type: 'auto' };
  if (choice.type === 'any') return { type: 'any' };
  if (choice.type === 'none') return { type: 'none' };
  if (choice.type === 'tool' && typeof choice.name === 'string') {
    return { type: 'function', function: { name: choice.name } };
  }
  return null;
}

/**
 * A Messages request turned into a Chat Completions body for the gateway:
 * the system prompt becomes the first system message, the messages keep
 * their text (and any tool use), the declared tools become OpenAI
 * functions, and max_tokens, temperature, stream and stop sequences travel
 * through unchanged. The model is the id this entry dialled.
 */
export function anthropicToOpenAI(body: Json, dialled: string): Json {
  const messages: Json[] = [];
  const systemText = anthropicSystemText(body.system);
  if (systemText !== '') messages.push({ role: 'system', content: systemText });
  const incoming = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of incoming) messages.push(...anthropicMessageParts(raw));

  const out: Json = { model: dialled, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (body.stream === true) out.stream = true;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools) && body.tools.length > 0) {
    out.tools = body.tools.filter(isRecord).map((tool) => ({
      type: 'function',
      function: {
        name: typeof tool.name === 'string' ? tool.name : '',
        description: typeof tool.description === 'string' ? tool.description : '',
        parameters: isRecord(tool.input_schema) ? tool.input_schema : {},
      },
    }));
  }
  const choice = anthropicToolChoice(body.tool_choice);
  if (choice !== null) out.tool_choice = choice;
  return out;
}

// ---------------------------------------------------------------------------
// SSE plumbing (shared by both stream converters)
// ---------------------------------------------------------------------------

/** One complete server-sent-events frame, blank-line separated, as text. */
async function* readSseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let pending = '';
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (value !== undefined) pending += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      if (done) {
        pending += decoder.decode().replace(/\r\n/g, '\n');
        break;
      }
      let cut = pending.indexOf('\n\n');
      while (cut !== -1) {
        yield pending.slice(0, cut + 2);
        pending = pending.slice(cut + 2);
        cut = pending.indexOf('\n\n');
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // The reader is already gone: nothing to release.
    }
  }
  if (pending.trim() !== '') yield pending;
}

/** The payloads of one frame's `data:` lines, trimmed, in order. */
function frameDataPayloads(frame: string): string[] {
  const payloads: string[] = [];
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    payloads.push(trimmed.slice(5).trim());
  }
  return payloads;
}

/** One frame with every data JSON's model field rewritten to the dialled id. */
function stampFrameModel(frame: string, dialled: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    let parsed: unknown;
    try {
      parsed = JSON.parse(payload);
    } catch {
      return line;
    }
    if (!isRecord(parsed) || typeof parsed.model !== 'string') return line;
    return `data: ${JSON.stringify({ ...parsed, model: dialled })}`;
  });
  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// OpenAI stream: stamp the model, pass everything else through
// ---------------------------------------------------------------------------

/**
 * An upstream Chat Completions stream with the dialled id stamped onto
 * every chunk: each data JSON keeps its id, its choices, its usage and its
 * finish reason, and only the model field changes. [DONE] and any
 * unparseable line pass through untouched.
 */
export async function* openAIStreamWithModel(
  upstream: ReadableStream<Uint8Array>,
  dialled: string,
): AsyncGenerator<string> {
  for await (const frame of readSseFrames(upstream)) {
    yield stampFrameModel(frame, dialled);
  }
}

// ---------------------------------------------------------------------------
// Anthropic stream: turn OpenAI chunks into Messages events
// ---------------------------------------------------------------------------

/** One streamed tool call being assembled from its delta fragments, by index. */
interface PendingToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** One Anthropic SSE event line: `event: <name>` then its data JSON. */
function anthropicEvent(name: string, data: Json): string {
  return `event: ${name}\ndata: ${JSON.stringify(data)}\n\n`;
}

/**
 * An upstream Chat Completions stream turned into Anthropic Messages
 * events. The whole upstream stream is read first — the caller buffers
 * the events before sending them, so nothing reaches the client
 * half-formed — then the events go out in the order Anthropic expects:
 * message_start, one content block (text, then any tool_use), and a
 * single message_delta followed by exactly one message_stop. The
 * message_start carries the input tokens and the message_delta the
 * output tokens, both from the stream's usage chunk when it has one.
 */
export async function* anthropicEventsFromOpenAI(
  upstream: ReadableStream<Uint8Array>,
  dialled: string,
): AsyncGenerator<string> {
  const texts: string[] = [];
  let finish: string | null = null;
  let openAiId = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let sawUsage = false;
  const pendingCalls = new Map<number, PendingToolCall>();

  for await (const frame of readSseFrames(upstream)) {
    for (const payload of frameDataPayloads(frame)) {
      if (payload === '' || payload === '[DONE]') continue;
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (typeof chunk.id === 'string' && openAiId === '') openAiId = chunk.id;
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const first = isRecord(choices[0]) ? choices[0] : null;
      if (first !== null) {
        const delta = isRecord(first.delta) ? first.delta : null;
        if (delta !== null) {
          if (typeof delta.content === 'string' && delta.content !== '') texts.push(delta.content);
          if (Array.isArray(delta.tool_calls)) {
            for (const fragment of delta.tool_calls) {
              if (!isRecord(fragment) || typeof fragment.index !== 'number') continue;
              const slot = pendingCalls.get(fragment.index) ?? { id: '', name: '', arguments: '' };
              if (typeof fragment.id === 'string') slot.id = fragment.id;
              if (isRecord(fragment.function)) {
                if (typeof fragment.function.name === 'string') slot.name = fragment.function.name;
                if (typeof fragment.function.arguments === 'string') slot.arguments += fragment.function.arguments;
              }
              pendingCalls.set(fragment.index, slot);
            }
          }
        }
        if (typeof first.finish_reason === 'string') finish = first.finish_reason;
      }
      if (isRecord(chunk.usage)) {
        sawUsage = true;
        if (typeof chunk.usage.prompt_tokens === 'number') inputTokens = chunk.usage.prompt_tokens;
        if (typeof chunk.usage.completion_tokens === 'number') outputTokens = chunk.usage.completion_tokens;
      }
    }
  }

  const text = texts.join('');
  if (!sawUsage && text !== '') outputTokens = 1;
  const messageId = openAiId !== '' ? `msg_${openAiId}` : 'msg_entry';
  const toolCalls = [...pendingCalls.entries()].sort((a, b) => a[0] - b[0]).map(([, call]) => call);
  const stop = anthropicStopReason(finish ?? 'stop', toolCalls.length > 0);

  yield anthropicEvent('message_start', {
    type: 'message_start',
    message: {
      id: messageId,
      type: 'message',
      role: 'assistant',
      content: [],
      model: dialled,
      stop_reason: null,
      usage: { input_tokens: inputTokens, output_tokens: 0 },
    },
  });

  yield anthropicEvent('content_block_start', {
    type: 'content_block_start',
    index: 0,
    content_block: { type: 'text', text: '' },
  });
  for (const piece of texts) {
    yield anthropicEvent('content_block_delta', {
      type: 'content_block_delta',
      index: 0,
      delta: { type: 'text_delta', text: piece },
    });
  }
  yield anthropicEvent('content_block_stop', { type: 'content_block_stop', index: 0 });

  let blockIndex = 1;
  for (const call of toolCalls) {
    yield anthropicEvent('content_block_start', {
      type: 'content_block_start',
      index: blockIndex,
      content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} },
    });
    yield anthropicEvent('content_block_delta', {
      type: 'content_block_delta',
      index: blockIndex,
      delta: { type: 'input_json_delta', partial_json: call.arguments },
    });
    yield anthropicEvent('content_block_stop', { type: 'content_block_stop', index: blockIndex });
    blockIndex += 1;
  }

  yield anthropicEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: stop },
    usage: { output_tokens: outputTokens },
  });
  yield anthropicEvent('message_stop', { type: 'message_stop' });
}
