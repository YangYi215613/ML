# MCP如何进行开发  

下面给你一份**MCP（Model Context Protocol）开发速通指南**，涵盖概念→脚手架→最小可用示例→接入客户端→调试与安全要点。我用**TypeScript**和**Python**各给一个最小例子，你可任选其一上手。

# 1) 开发前要点（1 分钟读懂）

- **MCP 是什么**：一种基于 **JSON-RPC** 的开放协议，让“客户端/宿主”（如 Claude Desktop、VS Code、Cursor 等）通过**工具（tools）**、**资源（resources）**、**提示（prompts）**与**服务器（server）**交互。([Anthropic](https://docs.anthropic.com/en/docs/mcp?utm_source=chatgpt.com))
- **传输层**：常用 `stdio`（本机可执行程序）或 WebSocket/HTTP(SSE)。开发时用 `stdio` 最简单。([Model Context Protocol](https://modelcontextprotocol.io/docs/concepts/architecture?utm_source=chatgpt.com))
- **官方文档 & SDK**：
  - 文档与快速上手（强烈推荐）：modelcontextprotocol.io（含 quickstart、连接到本地服务器指南）。([Model Context Protocol](https://modelcontextprotocol.io/quickstart/server?utm_source=chatgpt.com))
  - Python SDK（官方）：`modelcontextprotocol/python-sdk`。([GitHub](https://github.com/modelcontextprotocol/python-sdk?utm_source=chatgpt.com))

# 2) 最小可用示例

## 2.1 TypeScript 版本（基于官方思路）

1. 初始化项目

```bash
mkdir mcp-ts-demo && cd mcp-ts-demo
npm init -y
npm i @modelcontextprotocol/sdk typescript ts-node
npx tsc --init
```

1. 新建 `server.ts`（暴露一个简单工具 `echo`）

```ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";

const server = new Server(
  { name: "mcp-ts-demo", version: "0.1.0" },
  { capabilities: { tools: {} } }
);

server.tool("echo", {
  description: "Echo back your text",
  inputSchema: {
    type: "object",
    properties: { text: { type: "string" } },
    required: ["text"]
  },
  async handler({ text }) {
    return { content: [{ type: "text", text: `You said: ${text}` }] };
  }
});

await server.connect(new StdioServerTransport());
```

1. 在 `package.json` 里加启动脚本

```json
"scripts": {
  "start": "ts-node server.ts"
}
```

> 该结构与官方 quickstart 思路一致：使用 SDK 注册工具 → 通过 `stdio` 启动。([Model Context Protocol](https://modelcontextprotocol.io/quickstart/server?utm_source=chatgpt.com))

## 2.2 Python 版本（最精简）

1. 创建环境并安装

```bash
python -m venv .venv && source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
pip install modelcontextprotocol
```

1. 新建 `server.py`

```python
from modelcontextprotocol.server import Server
from modelcontextprotocol.transport.stdio import StdioServerTransport

server = Server({"name": "mcp-py-demo", "version": "0.1.0"},
                capabilities={"tools": {}})

@server.tool("echo", description="Echo back your text",
             input_schema={"type": "object",
                           "properties": {"text": {"type": "string"}},
                           "required": ["text"]})
async def echo_tool(args):
    return {"content": [{"type": "text", "text": f"You said: {args['text']}"}]}

async def main():
    await server.connect(StdioServerTransport())

if __name__ == "__main__":
    import asyncio; asyncio.run(main())
```

> 该 Python SDK实现了完整 MCP 规范，内置 stdio/SSE/HTTP 等传输，适合快速起步。([GitHub](https://github.com/modelcontextprotocol/python-sdk?utm_source=chatgpt.com))

# 3) 把服务器接入宿主（以 Claude Desktop 为例）

1. 打开官方指南“连接本地 MCP 服务器”，在配置里为你的服务器添加一段：
    （Mac: `~/Library/Application Support/Claude/claude_desktop_config.json`，Windows 路径以官方文档为准）

```json
{
  "mcpServers": {
    "mcp-ts-demo": {
      "command": "npm",
      "args": ["run", "start"],
      "env": {}
    },
    "mcp-py-demo": {
      "command": "python",
      "args": ["server.py"],
      "env": {}
    }
  }
}
```

1. 重启 Claude Desktop，在“工具/服务器”列表里即可看到并调用 `echo`。

> 具体路径和操作以官方“Connect to local servers”文档为准。([Model Context Protocol](https://modelcontextprotocol.io/docs/develop/connect-local-servers?utm_source=chatgpt.com))

（如果你用 **VS Code** 的原生 MCP 支持，同样可在其指南中接入自建 MCP 服务器。）([Visual Studio Code](https://code.visualstudio.com/api/extension-guides/ai/mcp?utm_source=chatgpt.com))

# 4) 进阶：除 Tools 外，还可以暴露……

- **Resources**：如只读/可检索的文件、数据库结果、URL 内容等。
- **Prompts**：为宿主提供可复用 Prompt 模板。
- **多传输支持**：除 `stdio` 外，可用 WebSocket/HTTP（SSE）。

> 这些都是 MCP 的“核心原语”，详见架构与概念文档。([Model Context Protocol](https://modelcontextprotocol.io/docs/concepts/architecture?utm_source=chatgpt.com))

# 5) 调试与常见问题

- **本地调试**：先在终端执行 `npm run start` 或 `python server.py`，确保进程能常驻、无异常输出，再接入宿主。
- **无法被宿主识别**：检查可执行命令、工作目录、环境变量、JSON 配置拼写与路径。
- **工具入参与返回**：严格遵守 JSON Schema 与 MCP 的内容格式（`{ content: [{type:"text", text:"..."}] }`）。官方 quickstart 有更完整范例。([Model Context Protocol](https://modelcontextprotocol.io/quickstart/server?utm_source=chatgpt.com))
- **Claude Code 也可当 MCP 服务器**（反向用法），用于对接到其它宿主。([Anthropic](https://docs.anthropic.com/en/docs/claude-code/mcp?utm_source=chatgpt.com))

# 6) 安全与产品化建议

- **最小权限**：工具只暴露必要能力；对文件系统/网络访问做白名单与路径限制。
- **参数校验**：对工具输入做严格 schema 校验与类型检查，避免命令注入/SSRF。
- **可观测性**：为每次 `tools/call` 记录参数、耗时与错误。
- **用户确认**：涉及外部副作用（写文件、发请求、Git 推送）前要求宿主侧二次确认（多数宿主已有交互弹窗机制）。
- **跟进生态**：Windows/VS Code 等正原生支持 MCP，生态在快速演进，可关注最新指南与安全建议。([The Verge](https://www.theverge.com/news/669298/microsoft-windows-ai-foundry-mcp-support?utm_source=chatgpt.com))

------











