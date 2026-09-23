const { AWSBedrockLLM } = require("../../../../utils/AiProviders/bedrock");
const {
  openaiBaseURL,
} = require("../../../../utils/AiProviders/bedrock/endpoints");
const AWSBedrockProvider = require("../../../../utils/agents/aibitat/providers/bedrock.js");
const { OpenAI } = require("openai");
const { getCustomModels } = require("../../../../utils/helpers/customModels");

const ORIGINAL_ENV = process.env;

beforeEach(() => {
  process.env = {
    ...ORIGINAL_ENV,
    AWS_BEDROCK_LLM_API_KEY: "test-key",
    AWS_BEDROCK_LLM_REGION: "eu-west-1",
  };
});

afterEach(() => {
  process.env = ORIGINAL_ENV;
  jest.restoreAllMocks();
});

describe("openaiBaseURL", () => {
  it.each([
    [undefined, "https://bedrock-mantle.eu-west-1.api.aws/v1"],
    ["openai.gpt-oss-120b", "https://bedrock-mantle.eu-west-1.api.aws/v1"],
    ["qwen.qwen3-32b", "https://bedrock-mantle.eu-west-1.api.aws/v1"],
    ["openai.gpt-6-sol", "https://bedrock-mantle.eu-west-1.api.aws/openai/v1"],
    ["openai.gpt-5.5", "https://bedrock-mantle.eu-west-1.api.aws/openai/v1"],
    [
      "us.openai.gpt-6-luna",
      "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1",
    ],
    [
      "global.openai.gpt-6-sol",
      "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1",
    ],
  ])("routes %s to %s", (model, url) => {
    expect(openaiBaseURL("eu-west-1", model)).toBe(url);
  });

  it("is used by the chat and agent clients", () => {
    const model = "global.openai.gpt-6-sol";
    const url = "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1";
    expect(new AWSBedrockLLM(null, model).openai.baseURL).toBe(url);
    expect(new AWSBedrockProvider({ model }).client.baseURL).toBe(url);
  });
});

describe("OpenAI GPT request params", () => {
  const messages = [{ role: "user", content: "hi" }];
  const tools = [
    {
      name: "get_weather",
      description: "Get the weather",
      parameters: { type: "object", properties: {} },
    },
  ];

  function fakeCreate() {
    return jest.fn(async ({ stream }) => {
      if (!stream)
        return {
          choices: [{ message: { role: "assistant", content: "ok" } }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        };
      return (async function* () {
        yield {
          choices: [{ delta: { content: "ok" }, finish_reason: "stop" }],
        };
      })();
    });
  }

  function chatLLM(model, create) {
    const llm = new AWSBedrockLLM(null, model);
    llm.openai = { chat: { completions: { create } } };
    return llm;
  }

  function agent(model, create) {
    const provider = new AWSBedrockProvider({ model });
    provider._client = { chat: { completions: { create } } };
    return provider;
  }

  const chatCallSites = [
    ["chat", (m, c) => chatLLM(m, c).getChatCompletion(messages, {})],
    [
      "chat stream",
      (m, c) => chatLLM(m, c).streamGetChatCompletion(messages, {}),
    ],
  ];
  const agentCallSites = [
    ["agent complete", (m, c) => agent(m, c).complete(messages, tools)],
    ["agent stream", (m, c) => agent(m, c).stream(messages, tools, () => {})],
  ];

  describe.each([
    "openai.gpt-6-sol",
    "us.openai.gpt-6-luna",
    "openai.gpt-5.6-sol",
  ])("%s", (model) => {
    it.each(chatCallSites)("%s omits temperature", async (_label, run) => {
      const create = fakeCreate();
      await run(model, create);
      expect(create.mock.calls[0][0].temperature).toBeUndefined();
    });

    it.each(agentCallSites)(
      "%s sends tools with reasoning_effort none",
      async (_label, run) => {
        const create = fakeCreate();
        await run(model, create);
        expect(create.mock.calls[0][0].tools).toHaveLength(1);
        expect(create.mock.calls[0][0].reasoning_effort).toBe("none");
      }
    );
  });

  describe.each(["openai.gpt-oss-120b", "qwen.qwen3-32b"])("%s", (model) => {
    it.each(chatCallSites)(
      "%s keeps the default temperature",
      async (_label, run) => {
        const create = fakeCreate();
        await run(model, create);
        expect(create.mock.calls[0][0].temperature).toBe(0.7);
      }
    );

    it.each(agentCallSites)(
      "%s does not send reasoning_effort",
      async (_label, run) => {
        const create = fakeCreate();
        await run(model, create);
        expect(create.mock.calls[0][0]).not.toHaveProperty("reasoning_effort");
      }
    );
  });
});

describe("Bedrock model picker", () => {
  it("lists OpenAI GPT inference profiles next to Anthropic ones", async () => {
    jest.spyOn(console, "error").mockImplementation(() => {});
    jest
      .spyOn(OpenAI.Models.prototype, "list")
      .mockRejectedValue(new Error("no catalog"));
    jest.spyOn(global, "fetch").mockResolvedValue({
      ok: true,
      json: async () => ({
        inferenceProfileSummaries: [
          "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
          "global.openai.gpt-6-sol",
          "us.meta.llama3-3-70b-instruct-v1:0",
        ].map((inferenceProfileId) => ({
          inferenceProfileId,
          status: "ACTIVE",
        })),
      }),
    });

    const { models } = await getCustomModels("bedrock", "test-key", null, {
      region: "eu-west-1",
    });
    expect(models.map((model) => model.id)).toEqual([
      "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
      "global.openai.gpt-6-sol",
    ]);
  });
});
