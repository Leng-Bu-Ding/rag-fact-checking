# 运行、验证与故障排查手册

最后核验：2026-08-10

所有命令从仓库根目录运行，也就是包含 `README.md`、`src/` 和 `scripts/` 的
目录。公开文档和代码配置始终使用相对路径。

## 1. 确认环境

```powershell
Get-Location
& '.\.conda\python.exe' --version
git status --short --branch
```

预期Python为3.11.x，Git分支为`main`或明确的功能分支。

## 2. 安装依赖

```powershell
& '.\.conda\python.exe' -m pip install -r requirements-dev.txt
```

这会向 `.conda/` 写入Python包，可能联网下载并占用磁盘；不会把依赖提交到Git。

## 3. 运行测试

```powershell
& '.\.conda\python.exe' -m pytest -q --basetemp '.\.test_tmp' -p no:cacheprovider
```

当前发布基线为75项测试通过。`.test_tmp*` 被Git忽略。

## 4. 启动Demo

```powershell
& '.\.conda\python.exe' '.\scripts\start_demo.py'
```

等待模型预热并显示ready后访问：

- 页面：http://127.0.0.1:8000
- API文档：http://127.0.0.1:8000/docs
- 健康检查：http://127.0.0.1:8000/health

首次运行可能从Hugging Face下载模型。后续通常读取缓存，但配置中的
`local_files_only: false` 允许联网检查缺失文件。

## 5. FinanceBench dry-run

```powershell
& '.\.conda\python.exe' '.\scripts\run_financebench_generation_evaluation.py' --dry-run
```

dry-run检查问题、Top-K证据、页码和Prompt，不发送API请求、不消耗Token费用、
不产生答案指标。

## 6. 配置真实API

只在当前PowerShell窗口设置：

```powershell
$env:RAG_API_BASE_URL = '<服务端点>'
$env:RAG_API_MODEL = '<模型标识>'
$env:RAG_API_KEY = '<密钥>'
```

也可写入被 Git 忽略的 `.env.local`。先运行 `--probe-api` 验证，再用 Dev 调试；
Prompt 冻结后才运行 Holdout 与全量：

```powershell
& '.\.conda\python.exe' '.\scripts\run_financebench_generation_evaluation.py' --probe-api
& '.\.conda\python.exe' '.\scripts\run_financebench_generation_evaluation.py' --evaluation-set dev
& '.\.conda\python.exe' '.\scripts\run_financebench_generation_evaluation.py' --evaluation-set holdout
& '.\.conda\python.exe' '.\scripts\run_financebench_generation_evaluation.py' --evaluation-set all
& '.\.conda\python.exe' '.\scripts\run_financebench_answer_judge.py' --evaluation-set all
```

不要把真实值粘贴到聊天、Notion、代码或Git。关闭终端后，以上会话级变量通常
消失。正式运行前先用1题或少量样本验证端点、模型名、返回结构和费用记录。

## 7. 常见问题

### `python`找不到

不要依赖全局PATH，继续使用：

```powershell
& '.\.conda\python.exe' ...
```

### 模型重新下载

检查Hugging Face缓存和本机`HF_HOME`。不要把本机绝对缓存路径提交到配置。

### API返回401/403

检查Key、账户权限、模型授权和Base URL；不要在错误日志中打印完整Key。

### CUDA Out of Memory

记录GPU型号、显存、batch、sequence length和量化设置；先减少负载做smoke test，
不要把失败静默吞掉或直接宣称某显存规格一定可行。

### 指标与README不一致

先核对 `results/public/*.json`，再更新README、Notion和简历。不要以手工抄写的
Notion表格反向覆盖真实结果。
