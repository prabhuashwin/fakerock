// Smoke test: AWS SDK v3 (JavaScript) against a running fakerock server.
import {
  BedrockRuntimeClient,
  ConverseCommand,
  ConverseStreamCommand,
  InvokeModelCommand,
  InvokeModelWithResponseStreamCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { NodeHttpHandler } from "@smithy/node-http-handler";

const URL_ = process.env.FAKEROCK_URL || "http://127.0.0.1:52999";
const STRICT_URL = process.env.FAKEROCK_STRICT_URL;
const MODEL = "anthropic.claude-test";
const REPLY = "Mock reply to: hi";
const MESSAGES = [{ role: "user", content: [{ text: "hi" }] }];
const INVOKE_BODY = JSON.stringify({
  anthropic_version: "bedrock-2023-05-31",
  max_tokens: 50,
  messages: [{ role: "user", content: "hi" }],
});

const mkClient = (endpoint) =>
  new BedrockRuntimeClient({
    region: "us-east-1",
    endpoint,
    credentials: { accessKeyId: "x", secretAccessKey: "x" },
    maxAttempts: 1,
    requestHandler: new NodeHttpHandler(), // fakerock speaks HTTP/1.1 only; SDK default is HTTP/2
  });
const client = mkClient(URL_);

async function post(base, path, body) {
  const r = await fetch(base + path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST ${path} -> ${r.status}`);
}
async function resetServer(base) {
  await post(base, "/_fakerock/modes", {});
  await post(base, "/_fakerock/reset");
}
function eq(actual, expected, what) {
  if (actual !== expected) {
    throw new Error(`${what}: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
  }
}
const decode = (bytes) => JSON.parse(new TextDecoder().decode(bytes));

async function rejects(p) {
  try {
    await p;
  } catch (e) {
    return e;
  }
  throw new Error("expected rejection, call succeeded");
}

const checks = [
  ["converse", async () => {
    const r = await client.send(new ConverseCommand({ modelId: MODEL, messages: MESSAGES }));
    eq(r.output.message.content[0].text, REPLY, "text");
  }],
  ["converse-stream", async () => {
    const r = await client.send(new ConverseStreamCommand({ modelId: MODEL, messages: MESSAGES }));
    let text = "";
    for await (const ev of r.stream) if (ev.contentBlockDelta) text += ev.contentBlockDelta.delta.text;
    eq(text, REPLY, "text");
  }],
  ["invoke-model", async () => {
    const r = await client.send(new InvokeModelCommand({
      modelId: MODEL, contentType: "application/json", accept: "application/json", body: INVOKE_BODY,
    }));
    eq(decode(r.body).content[0].text, REPLY, "text");
  }],
  ["invoke-model-with-response-stream", async () => {
    const r = await client.send(new InvokeModelWithResponseStreamCommand({
      modelId: MODEL, contentType: "application/json", body: INVOKE_BODY,
    }));
    let text = "";
    for await (const ev of r.body) {
      if (!ev.chunk) continue;
      const j = decode(ev.chunk.bytes);
      if (j.type === "content_block_delta") text += j.delta.text;
    }
    eq(text, REPLY, "text");
  }],
  ["mode throttle -> ThrottlingException 429", async () => {
    await post(URL_, "/_fakerock/modes", { chat: { _default: "throttle" } });
    const e = await rejects(client.send(new ConverseCommand({ modelId: MODEL, messages: MESSAGES })));
    eq(e.name, "ThrottlingException", "name");
    eq(e.$metadata.httpStatusCode, 429, "status");
  }],
  ["mode stream_throttle -> converse-stream error after delta", async () => {
    await post(URL_, "/_fakerock/modes", { chat: { _default: "stream_throttle" } });
    const r = await client.send(new ConverseStreamCommand({ modelId: MODEL, messages: MESSAGES }));
    let deltas = 0;
    const e = await rejects((async () => {
      for await (const ev of r.stream) if (ev.contentBlockDelta) deltas++;
    })());
    if (deltas < 1) throw new Error("no delta before error");
    eq(e.name, "ThrottlingException", "name");
  }],
  ["mode stream_error -> invoke stream error after chunk", async () => {
    await post(URL_, "/_fakerock/modes", { chat: { _default: "stream_error" } });
    const r = await client.send(new InvokeModelWithResponseStreamCommand({
      modelId: MODEL, contentType: "application/json", body: INVOKE_BODY,
    }));
    let chunks = 0;
    const e = await rejects((async () => {
      for await (const ev of r.body) if (ev.chunk) chunks++;
    })());
    if (chunks < 1) throw new Error("no chunk before error");
    eq(e.name, "ModelStreamErrorException", "name");
  }],
];

if (STRICT_URL) {
  checks.push(["strict: assistant-first -> ValidationException 400", async () => {
    const e = await rejects(mkClient(STRICT_URL).send(new ConverseCommand({
      modelId: MODEL, messages: [{ role: "assistant", content: [{ text: "hi" }] }],
    })));
    eq(e.name, "ValidationException", "name");
    eq(e.$metadata.httpStatusCode, 400, "status");
  }]);
} else {
  console.log("skip - strict check (FAKEROCK_STRICT_URL not set)");
}

for (const [name, fn] of checks) {
  try {
    await resetServer(URL_);
    await fn();
    console.log(`ok - ${name}`);
  } catch (e) {
    console.error(`FAIL - ${name}`);
    console.error(e);
    process.exit(1);
  }
}
await resetServer(URL_);
