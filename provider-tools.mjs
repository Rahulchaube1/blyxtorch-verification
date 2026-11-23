// Node host for the local BlyxTorch classifier. Provider credentials stay with the calling application.
import { execFile } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";
const execute = promisify(execFile);

export function localClassifier(executable, checkpoint) {
  const cli = path.resolve(executable);
  const model = path.resolve(checkpoint);
  async function run(args) {
    const { stdout } = await execute(cli, args, { timeout: 30_000, maxBuffer: 16 * 1024 * 1024 });
    return JSON.parse(stdout);
  }
  return {
    schema: (provider) => run(["probe-tool", provider]),
    async call(name, args) {
      if (name !== "classify_features" || !args || Object.keys(args).length !== 1 ||
          !Array.isArray(args.features) || !args.features.length || !args.features.every(Number.isFinite)) {
        throw new TypeError("Expected classify_features with one finite numeric array");
      }
      // Pass vectors by file so wide embeddings also work on Windows.
      const directory = await mkdtemp(path.join(tmpdir(), "blyxtorch-features-"));
      try {
        const row = path.join(directory, "row.csv");
        await writeFile(row, args.features.join(","), { mode: 0o600 });
        const result = await run(["probe-predict-file", model, row]);
        if (!Number.isInteger(result.class_index)) throw new Error("Expected a classification checkpoint");
        return result;
      } finally {
        await rm(directory, { recursive: true, force: true });
      }
    },
  };
}

// The application chooses a model it can access and supplies an initialized SDK client.
export async function openAIClassification(client, model, prompt, classifier) {
  const tools = [await classifier.schema("openai")];
  const input = [{ role: "user", content: prompt }];
  const first = await client.responses.create({ model, tools, input, store: false, include: ["reasoning.encrypted_content"],
    tool_choice: { type: "function", name: "classify_features" } });
  const results = [];
  for (const item of first.output.filter((part) => part.type === "function_call")) {
    const value = await classifier.call(item.name, JSON.parse(item.arguments));
    results.push({ type: "function_call_output", call_id: item.call_id, output: JSON.stringify(value) });
  }
  if (!results.length) return first;
  return client.responses.create({ model, store: false, tools, tool_choice: "none", input: [...input, ...first.output, ...results] });
}

export async function claudeClassification(client, model, prompt, classifier) {
  const tools = [await classifier.schema("claude")];
  const messages = [{ role: "user", content: prompt }];
  const first = await client.messages.create({ model, max_tokens: 512, tools, messages });
  const results = [];
  for (const item of first.content.filter((block) => block.type === "tool_use")) {
    const value = await classifier.call(item.name, item.input);
    results.push({ type: "tool_result", tool_use_id: item.id, content: JSON.stringify(value) });
  }
  if (!results.length) return first;
  return client.messages.create({ model, max_tokens: 512, messages: [...messages,
    { role: "assistant", content: first.content }, { role: "user", content: results }], tools, tool_choice: { type: "none" } });
}
