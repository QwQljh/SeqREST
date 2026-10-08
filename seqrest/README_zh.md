# SeqREST

SeqREST 使用本地 LLM 生成 REST API 请求序列，复用响应中的资源，并进行序列与参数变异。支持 OpenAPI JSON 和 YAML。

## 快速使用

1. 将本目录放入已有 RESTGym 的 `tools/seqrest`。
2. 启动本地 OpenAI 兼容模型服务，在 `model-config.json` 中修改 `LLM_BASE_URL` 和实际模型名称 `LLM_MODEL`。服务启用认证时，填写可选的 `LLM_API_KEY`；否则保留 `EMPTY`。显式传入容器的环境变量优先于文件。工具不会启动模型服务或下载权重，地址必须能从工具容器访问。
3. 将 `restgym-tool-config.yml` 改为 `enabled: true`。只测试 SeqREST 时，关闭其他工具。
4. 在 RESTGym 根目录运行 `./restgym.sh b`，选择构建工具镜像。镜像名为 `restgym-seqrest`；已有镜像可能被跳过，修改源码或配置后应重新构建。
5. 在 RESTGym 中选择 API、设置测试时间，再运行 `./restgym.sh l`。

启动测试前，检查模型能否正常生成回复。地址、模型名和密钥需与 `model-config.json` 一致；使用真实密钥时替换 `Bearer EMPTY`。成功响应的 `choices[0].message.content` 应包含非空回复：

```bash
curl -fsS --max-time 60 http://127.0.0.1:11434/v1/chat/completions -H 'Content-Type: application/json' -H 'Authorization: Bearer EMPTY' -d '{"model":"qwen3-14b","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

需要明确重新构建时，在 RESTGym 根目录执行：

```bash
docker build -t restgym-seqrest -f tools/seqrest/Dockerfile .
```

## 文件与输出

`src/seqrest` 是运行源码，`run.sh` 自动启动 `python -m seqrest`。唯一依赖文件为 `requirements.txt`，包含经过测试的完整依赖及精确版本。`runtime-defaults.json` 保存内部默认值，正常使用只需修改模型配置并启用工具。

RESTGym 提供 `API`、`HOST`、`PORT`、`TOOL`、`RUN` 和以分钟计的 `TIME_BUDGET`，并收集 HTTP 交互与工具日志。默认不额外保存容器内部文件（`SEQREST_SAVE_ARTIFACTS=false`）。

模型的实际上下文容量由服务端决定，内部 `LLM_CONTEXT_WINDOW` 只是提示词预算假设。OpenAPI 外部 `$ref` 文件需预先打包。

[English](README.md)

公开提交包保留 `EMPTY`。真实密钥写入配置后会进入构建的镜像；运行环境允许时，可显式传入容器环境变量，避免将密钥嵌入镜像。
