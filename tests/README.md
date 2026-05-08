# AI 小益 Agent Core 单元测试

本项目包含完整、可直接运行的单元测试，覆盖所有核心功能。

## 测试模块

### 1. `test_core.py` - Agent Core 测试

测试 Agent Core 核心功能，包括：

- **任务拆解测试** (`TestTaskDecompose`)
  - `test_task_decompose_knowledge_query` - 知识查询意图任务拆解
  - `test_task_decompose_data_query` - 数据查询意图任务拆解
  - `test_task_decompose_task_operation` - 任务操作意图任务拆解
  - `test_task_decompose_unknown_intent` - 未知意图任务拆解
  - `test_task_decompose_node` - 任务拆解节点测试

- **工具执行测试** (`TestToolExecute`)
  - `test_execute_rag_search_success` - RAG 搜索成功执行
  - `test_execute_sql_task_success` - SQL 任务成功执行
  - `test_execute_echo_task_success` - Echo 任务成功执行
  - `test_execute_unknown_task_type` - 未知任务类型处理
  - `test_tool_execute_with_retry` - 工具执行重试机制

- **结果聚合测试** (`TestResultAggregate`)
  - `test_result_aggregate_with_successful_results` - 成功结果聚合
  - `test_result_aggregate_with_failed_results` - 失败结果聚合
  - `test_result_aggregate_with_empty_results` - 空结果聚合
  - `test_result_aggregate_mixed_results` - 混合结果聚合

- **完整流程测试** (`TestAgentWorkflow`)
  - `test_full_workflow_knowledge_query` - 知识查询完整流程
  - `test_full_workflow_data_query` - 数据查询完整流程
  - `test_stream_workflow` - 流式执行流程
  - `test_workflow_with_error` - 错误流程处理
  - `test_workflow_max_steps_limit` - 最大步数限制

- **错误处理测试** (`TestErrorHandling`)
  - `test_friendly_error_not_found` - NotFound 错误友好提示
  - `test_friendly_error_timeout` - Timeout 错误友好提示
  - `test_friendly_error_permission` - Permission 错误友好提示
  - `test_friendly_error_validation` - Validation 错误友好提示
  - `test_friendly_error_unknown` - 未知错误友好提示
  - `test_error_handle_node_retry` - 错误处理重试
  - `test_error_handle_node_max_retry` - 最大重试次数

### 2. `test_memory.py` - 记忆系统测试

测试三级记忆系统，包括：

- **短期记忆测试** (`TestShortMemory`)
  - `test_add_episodic_memory` - 添加短期记忆
  - `test_get_episodic_memory` - 获取短期记忆
  - `test_episodic_memory_capacity_limit` - 短期记忆容量限制
  - `test_clear_episodic_memory` - 清除短期记忆
  - `test_episodic_memory_with_metadata` - 带元数据的短期记忆

- **中期记忆测试** (`TestMediumMemory`)
  - `test_add_semantic_memory` - 添加语义记忆
  - `test_get_semantic_memory` - 获取语义记忆
  - `test_semantic_memory_capacity_limit` - 语义记忆容量限制
  - `test_semantic_memory_multiple_keys` - 多键语义记忆
  - `test_clear_semantic_memory_specific_key` - 清除特定键记忆
  - `test_clear_all_semantic_memory` - 清除所有语义记忆
  - `test_semantic_memory_with_tags` - 带标签的语义记忆

- **长期记忆测试** (`TestLongMemory`)
  - `test_add_long_term_memory` - 添加长期记忆
  - `test_get_long_term_memory_without_query` - 获取长期记忆（无查询）
  - `test_long_term_memory_capacity_limit` - 长期记忆容量限制
  - `test_long_term_memory_with_tags` - 带标签的长期记忆

- **Elasticsearch 集成测试** (`TestElasticsearchIntegration`)
  - `test_sync_to_es_success` - 成功同步到 ES
  - `test_sync_to_es_with_existing_index` - 同步到已存在的 ES 索引
  - `test_search_long_term_from_es` - 从 ES 检索长期记忆
  - `test_search_with_es_disconnected` - ES 断开时使用本地存储
  - `test_sync_to_es_failure` - ES 同步失败处理

### 3. `test_prompt.py` - 提示词模板测试

测试提示词管理器，包括：

- **基础功能测试** (`TestPromptManagerBasics`)
  - `test_initialization` - 初始化测试
  - `test_has_prompt` - 检查提示词存在
  - `test_list_prompts` - 列出所有提示词

- **提示词获取测试** (`TestGetPrompt`)
  - `test_get_existing_prompt` - 获取存在的提示词
  - `test_get_nonexistent_prompt` - 获取不存在的提示词
  - `test_get_prompt_no_formatting` - 不含变量的提示词

- **模板渲染测试** (`TestPromptTemplateRender`)
  - `test_task_decompose_template_render` - 任务拆解模板渲染
  - `test_result_aggregate_template_render` - 结果聚合模板渲染
  - `test_template_render_with_complex_variables` - 复杂变量渲染
  - `test_template_render_with_chinese_characters` - 中文字符渲染

- **错误处理测试** (`TestTemplateRenderErrors`)
  - `test_missing_template_variable` - 缺少模板变量
  - `test_template_formatting_error` - 模板格式化错误
  - `test_template_with_empty_variable_name` - 空变量名

## 运行测试

### 方式 1: 使用 pytest 直接运行

```bash
# 运行所有测试
pytest -v tests/

# 运行特定模块测试
pytest -v tests/test_core.py
pytest -v tests/test_memory.py
pytest -v tests/test_prompt.py

# 运行特定测试
pytest -v tests/test_core.py::TestTaskDecompose::test_task_decompose_knowledge_query

# 生成覆盖率报告
pytest --cov=agent --cov-report=html tests/
```

### 方式 2: 使用运行脚本

```bash
# 运行所有测试
python tests/run_tests.py

# 运行指定模块
python tests/run_tests.py --module core
python tests/run_tests.py --module memory
python tests/run_tests.py --module prompt

# 生成覆盖率报告
python tests/run_tests.py --coverage

# 生成 HTML 测试报告
python tests/run_tests.py --html-report

# 按标记运行测试
python tests/run_tests.py --marks "unit"

# 列出所有测试
python tests/run_tests.py --list
```

### 方式 3: 从项目根目录运行

```bash
# 激活虚拟环境后
pytest

# 或
python -m pytest
```

## 测试特性

- ✅ **独立运行**：每个测试可独立运行，不依赖外部服务
- ✅ **Mock 外部依赖**：使用 Mock 模拟 Redis、ES、MCP 等外部服务
- ✅ **覆盖正常流程**：测试所有正常的业务流程
- ✅ **覆盖异常流程**：测试错误处理和异常情况
- ✅ **工具重试测试**：覆盖工具执行的重试机制
- ✅ **最大步数限制**：测试执行步数限制
- ✅ **错误处理**：测试各种错误的友好提示

## 测试覆盖率

运行 `pytest --cov=agent --cov-report=html` 后，查看 `htmlcov/index.html` 获取详细覆盖率报告。

预期覆盖率：
- `agent/core.py`: > 90%
- `agent/memory.py`: > 85%
- `agent/prompt.py`: > 95%

## 依赖安装

确保安装测试依赖：

```bash
pip install pytest pytest-cov pytest-html pytest-mock
```

或更新 `requirements.txt`:

```
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-html>=3.2.0
pytest-mock>=3.11.1
```

## 验证项目完整可用

运行以下命令验证项目：

```bash
# 1. 运行所有测试（应该全部通过）
pytest -v tests/

# 2. 生成覆盖率报告（应该看到高覆盖率）
pytest --cov=agent --cov-report=term tests/

# 3. 测试项目主模块（可选）
python -c "from agent.core import agent_core; print('Agent Core 加载成功')"
python -c "from agent.memory import memory_system; print('Memory System 加载成功')"
python -c "from agent.prompt import prompt_manager; print('Prompt Manager 加载成功')"

# 4. 测试 MCP 模块（可选）
python -c "from mcp.scheduler import skill_scheduler; print('Skill Scheduler 加载成功')"
python -c "from mcp.protocol import MCPRequest, MCPResponse; print('MCP Protocol 加载成功')"
```

## 测试文件结构

```
tests/
├── __init__.py           # 测试包初始化
├── conftest.py           # pytest 配置和共享 fixtures
├── README.md             # 本文档
├── run_tests.py          # 测试运行脚本
├── test_core.py          # Agent Core 测试
├── test_memory.py        # 记忆系统测试
└── test_prompt.py        # 提示词模板测试
```

## 持续集成

在 CI/CD 流程中运行测试：

```yaml
# GitHub Actions 示例
- name: Run tests
  run: |
    pip install pytest pytest-cov
    pytest --cov=agent --cov-report=xml --junitxml=test-results.xml tests/

- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## 常见问题

### Q: 测试失败提示 "ImportError"

**A**: 确保在项目根目录运行测试，或设置正确的 PYTHONPATH：

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
pytest tests/
```

### Q: 测试通过但覆盖率低

**A**: 某些代码路径可能需要特殊条件才能触发。检查日志中的警告信息。

### Q: Mock 不工作

**A**: 确保正确使用 `patch` 装饰器或 fixture。参考 `conftest.py` 中的示例。

## 贡献指南

添加新测试时：

1. 在对应的测试文件中添加测试类和测试方法
2. 使用描述性的测试方法名
3. 测试正常流程和异常流程
4. 确保测试独立可运行
5. 更新本文档

## 联系方式

如有问题，请联系项目维护者或在 GitHub Issues 中报告。
