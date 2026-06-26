# Claude Code 接入 OpenAI 兼容 API 的适配服务 — 技术报告

## 1. 整体架构

```
Claude Code (客户端)
    │  POST /v1/messages   (Anthropic Messages API 格式)
    │  x-api-key: <用户的key>
    ▼
你的适配服务 (127.0.0.1:8765)
    │  ① 接收 Anthropic 格式请求
    │  ② 提取 API Key / 替换认证
    │  ③ Anthropic → OpenAI Chat Completions 格式转换
    │  ④ 转发到上游 API
    ▼
上游 API (如 Zen、Agnes、OpenRouter 等)
    │  POST /v1/chat/completions  (OpenAI 格式)
    │  Authorization: Bearer <上游key>
    ▼
你的适配服务
    │  ⑤ OpenAI → Anthropic 格式反向转换
    │  ⑥ 返回给 Claude Code
    ▼
Claude Code
```

## 2. Claude Code 发送的请求格式 (Anthropic Messages API)

### 2.1 请求

```http
POST /v1/messages HTTP/1.1
Host: 127.0.0.1:8765
Content-Type: application/json
x-api-key: <用户在 Claude Code 中配置的 key>
anthropic-version: 2023-06-01

{
  "model": "claude-haiku-4-5",       // Claude 模型名，需要映射
  "max_tokens": 4096,
  "system": "You are Claude...",      // 系统提示词 (可选)
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "..."}
  ],
  "tools": [                          // 工具定义 (可选)
    {
      "name": "read_file",
      "description": "Read a file",
      "input_schema": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"]
      }
    }
  ],
  "stream": true                      // 流式输出
}
```

### 2.2 Claude Code 在 Config 中的配置

用户在 Claude Code 的 `.claude.json` 或环境变量中设置：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8765",
    "ANTHROPIC_AUTH_TOKEN": "<用户填的API Key>",
    "ANTHROPIC_MODEL": "claude-haiku-4-5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-6",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-8"
  }
}
```

Claude Code 会用固定的 Claude 模型名（haiku/sonnet/opus）发请求，你的适配服务需要把这些模型名映射到上游的实际模型。

## 3. 格式转换：Anthropic → OpenAI Chat Completions

### 3.1 Request 转换

```python
def anthropic_to_openai(anthropic_body: dict) -> dict:
    """Anthropic Messages → OpenAI Chat Completions"""
    messages = []

    # 1. System prompt → 第一条消息的 system role
    if "system" in anthropic_body:
        # 去掉 Anthropic 可能插入的 billing header 文本
        system_text = anthropic_body["system"]
        if isinstance(system_text, str):
            system_text = strip_billing_header(system_text)
        messages.append({"role": "system", "content": system_text})

    # 2. 转换 messages
    for msg in anthropic_body.get("messages", []):
        role = msg["role"]  # user / assistant
        content = msg.get("content", "")

        # Anthropic content 可能是字符串或数组
        if isinstance(content, list):
            # 多模态内容：提取文本部分
            text_parts = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "image":
                    # 图片 base64 → OpenAI 格式
                    text_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block['source']['media_type']};base64,{block['source']['data']}"
                        }
                    })
            content = "\n".join(p for p in text_parts if isinstance(p, str)) or text_parts
        elif isinstance(content, str):
            pass  # 纯文本，直接使用

        messages.append({"role": role, "content": content})

    # 3. 处理 tool_use 消息
    # Claude Code 的 assistant 消息可能包含 tool_use blocks
    # 需要转成 OpenAI 的 tool_calls 格式
    messages = convert_tool_messages(messages, anthropic_body)

    result = {
        "model": map_model(anthropic_body.get("model", "")),  # 模型名映射
        "messages": messages,
    }

    # 4. max_tokens
    if "max_tokens" in anthropic_body:
        result["max_tokens"] = anthropic_body["max_tokens"]

    # 5. stream
    if anthropic_body.get("stream"):
        result["stream"] = True

    # 6. tools → OpenAI 格式
    if "tools" in anthropic_body:
        result["tools"] = convert_tools(anthropic_body["tools"])

    # 7. thinking → reasoning_effort (可选，某些模型支持)
    if "thinking" in anthropic_body:
        effort = resolve_reasoning_effort(anthropic_body["thinking"])
        if effort:
            result["reasoning_effort"] = effort

    return result
```

### 3.2 模型名映射

```python
# Claude Code 使用的固定模型名 → 上游实际模型名
MODEL_MAP = {
    "claude-haiku-4-5":   "deepseek-v4-flash-free",   # 免费/快速模型
    "claude-sonnet-4-6":  "deepseek-v4-pro",           # 中等模型
    "claude-opus-4-8":    "deepseek-v4-pro",           # 最强模型
    "claude-fable-5":     "deepseek-v4-pro",           # Fable 模型
    # 也支持直接传上游模型名
    "deepseek-v4-flash-free": "deepseek-v4-flash-free",
    "deepseek-v4-pro":        "deepseek-v4-pro",
}
```

### 3.3 Response 转换：OpenAI → Anthropic

```python
def openai_to_anthropic(openai_body: dict, original_model: str) -> dict:
    """OpenAI Chat Completions → Anthropic Messages"""
    choice = openai_body["choices"][0]
    message = choice["message"]

    content = []
    # 文本内容
    if "content" in message and message["content"]:
        content.append({"type": "text", "text": message["content"]})

    # Reasoning (for DeepSeek R1 style models)
    if "reasoning_content" in message and message["reasoning_content"]:
        content.insert(0, {
            "type": "thinking",
            "thinking": message["reasoning_content"]
        })

    # Tool calls → tool_use blocks
    if "tool_calls" in message:
        for tc in message["tool_calls"]:
            content.append({
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": json.loads(tc["function"]["arguments"])
            })

    result = {
        "id": openai_body.get("id", f"msg_{uuid4().hex[:24]}"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": original_model,  # 返回 Claude 模型名，保持一致
        "stop_reason": map_stop_reason(choice.get("finish_reason")),
        "usage": {
            "input_tokens": openai_body.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_body.get("usage", {}).get("completion_tokens", 0),
        }
    }
    return result
```

## 4. 流式处理 (SSE Streaming)

Claude Code 几乎总是用 `stream: true`。你的服务需要支持 SSE。

### 4.1 Anthropic SSE 格式 (请求 Claude Code 期望的)

```
event: message_start
data: {"type":"message_start","message":{"id":"...","type":"message","role":"assistant","model":"claude-haiku-4-5","usage":{"input_tokens":100}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50}}

event: message_stop
data: {"type":"message_stop"}
```

### 4.2 OpenAI SSE 格式 (上游返回的)

```
data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### 4.3 SSE 转换代码

```python
import json, uuid

def convert_openai_sse_to_anthropic(openai_stream, model: str, input_tokens: int = 0):
    """实时转换 OpenAI SSE → Anthropic SSE"""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    content_index = 0
    text_buffer = ""
    output_tokens = 0
    started = False

    for line in openai_stream:
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            # 发送结束事件
            yield format_sse("content_block_stop", {
                "type": "content_block_stop",
                "index": content_index
            })
            yield format_sse("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": output_tokens}
            })
            yield format_sse("message_stop", {"type": "message_stop"})
            break

        chunk = json.loads(data_str)
        delta = chunk["choices"][0].get("delta", {})
        finish = chunk["choices"][0].get("finish_reason")

        if not started:
            started = True
            yield format_sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": msg_id, "type": "message",
                    "role": "assistant", "model": model,
                    "usage": {"input_tokens": input_tokens}
                }
            })
            yield format_sse("content_block_start", {
                "type": "content_block_start",
                "index": content_index,
                "content_block": {"type": "text", "text": ""}
            })

        if "content" in delta and delta["content"]:
            text_buffer += delta["content"]
            output_tokens += 1
            yield format_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": content_index,
                "delta": {"type": "text_delta", "text": delta["content"]}
            })

def format_sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
```

## 5. 工具调用 (Tool Use) 处理

Claude Code 使用工具时的消息流：

```
user: "帮我读一下 README.md"
assistant: tool_use(name="read_file", input={"path": "README.md"})  ← 你的服务需要保留这个
user: tool_result(content="文件内容...")                              ← 直接透传
assistant: "README.md 的内容是..."                                    ← 正常回复
```

在 Anthropic ↔ OpenAI 转换时，tool_use 需要映射：

```python
# Anthropic tool_use → OpenAI tool_calls
def convert_tool_use_to_openai(content_blocks: list) -> dict:
    tool_calls = []
    for block in content_blocks:
        if block["type"] == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block["input"])
                }
            })
    return {"role": "assistant", "tool_calls": tool_calls} if tool_calls else None

# Anthropic tool_result → OpenAI tool message
def convert_tool_result_to_openai(content_blocks: list) -> dict:
    for block in content_blocks:
        if block["type"] == "tool_result":
            return {
                "role": "tool",
                "tool_call_id": block["tool_use_id"],
                "content": block["content"]
            }
    return None
```

## 6. 认证处理

Claude Code 发送 `x-api-key: <用户key>`。你的服务：

```python
AUTH_TOKEN = "sk-your-upstream-api-key"  # 上游 API 的真实 key

def handle_auth(request):
    # 方案 A：忽略客户端 key，使用固定的上游 key
    upstream_headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (compatible; cc-adapter/1.0)"
    }
    return upstream_headers

    # 方案 B：透传客户端 key（如果上游接受相同的 key）
    # client_key = request.headers.get("x-api-key", "")
    # return {"Authorization": f"Bearer {client_key}", ...}
```

**注意**：必须设置 `User-Agent` header，否则 Cloudflare（Zen 等用 CDN 的上游）会拦截返回 403 error 1010。

## 7. 完整示例 (Python/FastAPI)

```python
import httpx
import json
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

app = FastAPI()

# 配置
UPSTREAM_URL = "https://opencode.ai/zen/v1/chat/completions"
UPSTREAM_KEY = "sk-your-api-key"
USER_AGENT = "Mozilla/5.0 (compatible; cc-adapter/1.0)"
LISTEN_PORT = 8765

MODEL_MAP = {
    "claude-haiku-4-5":  "deepseek-v4-flash-free",
    "claude-haiku-4-5-20251001": "deepseek-v4-flash-free",
    "claude-sonnet-4-6": "deepseek-v4-pro",
    "claude-opus-4-8":   "deepseek-v4-pro",
    "claude-fable-5":    "deepseek-v4-pro",
}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    model = body.get("model", "")
    stream = body.get("stream", True)

    # 1. Anthropic → OpenAI 格式转换
    openai_body = anthropic_to_openai(body)

    # 2. 转发到上游
    headers = {
        "Authorization": f"Bearer {UPSTREAM_KEY}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    async with httpx.AsyncClient(timeout=300) as client:
        if stream:
            # 流式转发
            req = client.build_request("POST", UPSTREAM_URL,
                json=openai_body, headers=headers)
            response = await client.send(req, stream=True)
            return StreamingResponse(
                convert_stream(response, model),
                media_type="text/event-stream"
            )
        else:
            # 非流式
            resp = await client.post(UPSTREAM_URL,
                json=openai_body, headers=headers)
            anthropic_resp = openai_to_anthropic(resp.json(), model)
            return anthropic_resp

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=LISTEN_PORT)
```

## 8. 部署与接入 Claude Code

### 8.1 启动服务

```bash
pip install fastapi uvicorn httpx
python adapter.py
# 服务运行在 http://127.0.0.1:8765
```

### 8.2 配置 Claude Code

用户设置环境变量：

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8765"
export ANTHROPIC_AUTH_TOKEN="any-value"  # 会被你的服务替换
export ANTHROPIC_MODEL="claude-haiku-4-5"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-haiku-4-5"
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4-6"
export ANTHROPIC_DEFAULT_OPUS_MODEL="claude-opus-4-8"
```

或修改 `~/.claude.json`：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8765",
    "ANTHROPIC_AUTH_TOKEN": "any-value",
    "ANTHROPIC_MODEL": "claude-haiku-4-5"
  }
}
```

### 8.3 在 cc-switch 中配置

通过 cc-switch 数据库添加供应商：

```sql
INSERT INTO providers (id, name, app_type, settings_config, meta, is_current)
VALUES (
  'zen-custom', 'OpenCode Zen Custom', 'claude',
  '{"env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8765",
    "ANTHROPIC_AUTH_TOKEN": "any-value",
    "ANTHROPIC_MODEL": "deepseek-v4-flash-free",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash-free",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro"
  }}',
  '{"apiFormat": "openai_chat", "customUserAgent": "Mozilla/5.0 (compatible; cc-switch/1.0)"}',
  1
);
```

## 9. 关键注意事项

| 问题 | 原因 | 解决 |
|---|---|---|
| 502 Bad Gateway | 上游不可达 / 格式转换失败 | 检查 UPSTREAM_URL、认证 |
| 401 Unauthorized | API Key 无效 / 账户无余额 | 先用 curl 直接测试 key |
| 403 error 1010 | Cloudflare 拦截 (User-Agent) | **必须设 User-Agent header** |
| 429 Too Many Requests | 上游限流 | 加 backoff 重试逻辑 |
| 熔断器跳闸 | 连续失败被断路 | 重置 provider_health 表并重启 |
| 模型响应为空 | thinking 参数不兼容 | 移除 `thinking` / `reasoning_effort` 字段 |
| 工具调用丢失 | tool_use ↔ tool_calls 映射错误 | 正确转换 Anthropic tool_use 格式 |

## 10. 不同上游 API 端点参考

| 上游 | 端点 | 格式 |
|---|---|---|
| OpenCode Zen (Chat) | `https://opencode.ai/zen/v1/chat/completions` | OpenAI Chat |
| OpenCode Zen (Messages) | `https://opencode.ai/zen/v1/messages` | Anthropic 原生 |
| OpenCode Zen (Responses) | `https://opencode.ai/zen/v1/responses` | OpenAI Responses |
| DeepSeek 官方 | `https://api.deepseek.com/v1` | OpenAI Chat |
| OpenRouter | `https://openrouter.ai/api/v1` | OpenAI Chat |
| 通用 OpenAI 兼容 | `https://api.example.com/v1` | OpenAI Chat |

## 11. cc-switch 的 Provider-Specific Config 模式

你在 cc-switch 中看到的这个启动方式：

```
Using provider-specific claude config:
C:\Users\15775\AppData\Local\Temp\claude_49c40878-...-89bcc360e025_46100.json
```

### 11.1 工作原理

cc-switch 通过 Claude Code 的 `--settings` 参数注入提供商配置，不修改全局 `~/.claude.json`：

```
cc-switch (GUI)
    │
    │ ① 生成临时配置文件
    ▼
C:\Users\<user>\AppData\Local\Temp\claude_{provider_id}_{pid}.json
    │ 内容：
    │ {
    │   "env": {
    │     "ANTHROPIC_BASE_URL": "http://127.0.0.1:15722",
    │     "ANTHROPIC_AUTH_TOKEN": "<代理 API key>",
    │     "ANTHROPIC_MODEL": "claude-haiku-4-5",
    │     "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5",
    │     "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-6",
    │     "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-8"
    │   }
    │ }
    │
    │ ② 启动 Claude Code 并传入 --settings
    ▼
claude --settings "C:\Users\...\Temp\claude_{provider_id}_{pid}.json"
    │
    │ ③ Claude Code 读取临时配置，使用其中的 base_url 和 auth_token
    ▼
Claude Code → 127.0.0.1:15722 (cc-switch 代理) → 上游 API
```

### 11.2 Rust 源代码 (cc-switch)

```rust
// src/commands/misc.rs

fn launch_terminal_with_env(
    env_vars: Vec<(String, String)>,
    provider_id: &str,
    cwd: Option<&Path>,
) -> Result<(), String> {
    let temp_dir = std::env::temp_dir();
    let config_file = temp_dir.join(format!(
        "claude_{}_{}.json",
        provider_id,
        std::process::id()
    ));

    // 写入临时配置文件
    write_claude_config(&config_file, &env_vars)?;

    // Windows: 生成 bat 脚本并启动
    // 内容: claude --settings "config_file.json"
    launch_windows_terminal(&temp_dir, &config_file, cwd)
}

fn write_claude_config(
    config_file: &Path,
    env_vars: &[(String, String)],
) -> Result<(), String> {
    let mut config_obj = serde_json::Map::new();
    let mut env_obj = serde_json::Map::new();
    for (key, value) in env_vars {
        env_obj.insert(key.clone(), Value::String(value.clone()));
    }
    config_obj.insert("env".to_string(), Value::Object(env_obj));
    let json = serde_json::to_string_pretty(&config_obj)?;
    std::fs::write(config_file, json)
}
```

### 11.3 临时配置文件格式

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:15722",
    "ANTHROPIC_AUTH_TOKEN": "sk-xxx",
    "ANTHROPIC_MODEL": "claude-haiku-4-5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash-free",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro",
    "ANTHROPIC_DEFAULT_FABLE_MODEL": "deepseek-v4-pro",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "1000000"
  }
}
```

### 11.4 两种接入 Claude Code 的方式对比

| 方式 | 配置文件 | 影响范围 | 适用场景 |
|---|---|---|---|
| **全局 env** | `~/.claude.json` 的 `env` 字段 | 所有 Claude Code 会话 | 固定使用一个上游 |
| **Provider config** | `Temp\claude_{id}_{pid}.json` + `--settings` flag | 仅当前终端窗口 | cc-switch 切换供应商 |

### 11.5 自己实现 Provider Config 模式

如果你要写一个独立的适配服务，可以让用户这样启动：

```bash
# 生成临时配置
cat > /tmp/claude_zen.json << 'EOF'
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8765",
    "ANTHROPIC_AUTH_TOKEN": "any-value",
    "ANTHROPIC_MODEL": "claude-haiku-4-5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-6",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-8"
  }
}
EOF

# 启动 Claude Code
claude --settings /tmp/claude_zen.json
```

或者用环境变量（更简单）：

```bash
export ANTHROPIC_BASE_URL="http://127.0.0.1:8765"
export ANTHROPIC_AUTH_TOKEN="any-value"
export ANTHROPIC_MODEL="claude-haiku-4-5"
claude
```

## 12. 注入自定义 Agent 提示词

### 12.1 Claude Code CLI 原生支持 (推荐)

`claude --help` 中有这些直接注入提示词的 flag：

```bash
# 完整替换系统提示词
claude --system-prompt "你是一个专注于 Go 后端的架构师..."

# 追加到默认系统提示词后面
claude --append-system-prompt "本次会话始终使用中文回复"

# 从文件加载提示词
claude --system-prompt "$(cat my-prompt.md)"

# 定义自定义 agent（JSON 格式）
claude --agents '{"reviewer":{"description":"审查代码","prompt":"你是代码审查专家...","tools":["Read","Grep"]}}'

# 使用特定 agent
claude --agent reviewer

# 指定 settings 文件（cc-switch 的方式）
claude --settings /tmp/claude_zen_config.json
```

### 12.2 组合使用示例

```bash
# 一次性带自定义提示词 + adapter + agent 启动 Claude Code
export ANTHROPIC_BASE_URL="http://127.0.0.1:8765"
export ANTHROPIC_AUTH_TOKEN="any-value"

claude \
  --settings /tmp/claude_zen_config.json \
  --append-system-prompt "你正在使用 DeepSeek V4 Flash 免费版，请优先使用高效简洁的回复" \
  --agents '{"zen-expert":{"description":"Zen API 专家","prompt":"你是 Zen API 专家","tools":["Bash","Read","Grep"]}}' \
  --model haiku
```

### 12.3 在临时配置文件中注入提示词

cc-switch 用的 `--settings` JSON 文件也可以包含自定义提示词环境变量：

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://127.0.0.1:8765",
    "ANTHROPIC_AUTH_TOKEN": "any-value",
    "ANTHROPIC_MODEL": "claude-haiku-4-5"
  },
  "customSystemPrompt": "你是一个 Python 后端专家，使用 FastAPI 框架。",
  "agents": {
    "code-reviewer": {
      "description": "审查代码变更",
      "prompt": "你是一个代码审查专家...",
      "tools": ["Read", "Grep", "Bash"]
    }
  }
}
```

### 12.4 CLAUDE.md 自动注入（无干预方式）

Claude Code 启动时自动读取 `CLAUDE.md`，无需命令行参数。cc-switch 的 Prompt 管理就是操作这个文件：

```
~/.claude/CLAUDE.md          ← 全局，所有会话生效
<project>/CLAUDE.md          ← 项目级
<project>/.claude/CLAUDE.md  ← 项目本地 (不提交 git)
```

cc-switch 的 Prompt 管理代码就是简单的 CRUD + 写文件：

```rust
// prompt_files.rs
AppType::Claude => "CLAUDE.md"   // 文件名
AppType::Codex  => "AGENTS.md"   // Codex 用 AGENTS.md
// 路径: ~/.claude/CLAUDE.md

// services/prompt.rs - 启用提示词
fn enable_prompt(state, app, id) {
    let target_path = prompt_file_path(&app)?;  // ~/.claude/CLAUDE.md
    write_text_file(&target_path, &prompt.content)?;  // 写入内容
}
```

**结论**：Claude Code 的提示词完全由客户端处理。你的适配服务**不需要**在协议层做任何提示词注入——只需透传 `system` 字段。用户自行选择用哪种方式注入（CLAUDE.md / --system-prompt / --settings）。
