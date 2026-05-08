# AI小益 - 招商银行收益升级中台智能化运营助手

## 项目介绍

AI小益是一款基于大语言模型的智能化运营助手，专门服务于招商银行收益升级中台系统的业务人员。通过整合意图识别、知识库检索、SQL生成执行和任务调度等核心能力，AI小益能够将自然语言查询转换为结构化操作，帮助业务人员高效获取理财产品信息和数据。

### 核心特性

- **智能意图识别**：基于scikit-learn的TF-IDF + LogisticRegression模型，自动识别用户查询意图
- **混合知识检索**：基于Elasticsearch的向量相似度检索和BM25关键词检索，加权融合提升检索准确度
- **安全SQL生成**：基于GLM4.7大模型生成SQL查询，通过多层校验确保安全性
- **任务调度执行**：基于MCP协议的统一工具调度框架，支持同步和异步执行
- **三级记忆系统**：短期会话记忆、中期用户偏好记忆、长期全局记忆
- **流式输出**：提供实时的响应体验

### 适用场景

- 理财产品知识问答
- 产品数据查询和统计
- 业务操作任务执行
- 自然语言转SQL查询
- 业务规则和政策咨询

---

## 环境安装

### 前置要求

- Python 3.10+
- pip 21.0+

### 核心依赖

```bash
# 核心框架
pip install langchain==0.2.16 langchain-openai==0.1.23 langgraph==0.2.39

# 向量数据库
pip install elasticsearch==7.17.14

# 意图识别
pip install scikit-learn==1.3.2

# SQL校验
pip install sqlglot==23.8.0

# 工具与辅助
pip install python-dotenv==1.0.1 pydantic==2.9.2 redis==5.0.8
pip install numpy==1.26.4 pandas==2.2.3

# Token计数
pip install tiktoken
```

### 配置说明

在项目根目录创建`.env`文件，配置以下信息：

```env
# 火山引擎GLM4.7配置
OPENAI_API_BASE=https://ark.cn-beijing.volces.com/api/v3
OPENAI_API_KEY=你的火山引擎API密钥
LLM_MODEL=glm-4-7b-32k  # 或glm-4-7b-128k

# Elasticsearch配置
ES_HOST=http://localhost:9200
ES_USER=elastic
ES_PASSWORD=你的ES密码
ES_INDEX=finance_rag_knowledge_v1

# FastText意图模型路径
FASTTEXT_MODEL_PATH=./models/intent_model.bin

# 全局配置
MAX_RETRY_COUNT=2
MAX_EXECUTE_STEPS=3
CONTEXT_WINDOW_LIMIT=116000
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境

编辑`.env`文件，填入实际的API密钥和连接信息。

### 3. 运行程序

```bash
python main.py
```

### 4. 交互模式

启动后会进入交互模式，输入查询并获取响应：

```
AI小益> 什么是固收类理财产品？

AI小益正在处理...

固收类理财产品是指投资于固定收益类资产的理财产品，主要包括债券、存款、票据等。这类产品的风险相对较低，收益相对稳定。

AI小益>
```

### 5. 退出程序

输入 `quit` 或 `exit` 退出程序。

---

## 模块说明

### 核心配置模块

- **config.py**：全局配置单例类，管理大模型客户端、ES连接等

### Agent核心层

- **agent/core.py**：基于LangGraph的状态机，实现任务拆解、执行和结果聚合
- **agent/memory.py**：三级记忆系统（短期/中期/长期）
- **agent/prompt.py**：所有提示词模板

### 意图识别模块

- **intent/recognizer.py**：基于scikit-learn的意图识别器，支持自动冷启动训练

### RAG知识库模块

- **rag/retriever.py**：基于Elasticsearch的混合检索器，支持向量相似度和BM25关键词检索

### MCP协议与工具层

- **mcp/protocol.py**：MCP请求/响应格式定义
- **mcp/skill_base.py**：Skill基类，所有Skill必须继承
- **mcp/scheduler.py**：工具调度器，支持Skill注册和动态发现
- **mcp/skills/rag_skill.py**：RAG检索Skill
- **mcp/skills/echo_skill.py**：测试用Echo Skill
- **mcp/skills/sql_skill.py**：SQL生成与校验Skill

### CodeAct SQL模块

- **codeact/generator.py**：基于GLM4.7的SQL生成器
- **codeact/validator.py**：SQL校验器，包含AST语法校验和规则校验

### 工具函数

- **utils/token_counter.py**：Token计数工具
- **utils/desensitize.py**：数据脱敏工具
- **utils/logger.py**：日志工具

---

## 测试方法

### 1. 意图识别测试

```bash
python -c "from intent.recognizer import test_intent_recognizer; test_intent_recognizer()"
```

### 2. RAG检索 retriever测试

```bash
python -c "from rag.test_retriever import test_rag_retriever; test_rag_retriever()"
```

### 3. MCP协议测试

```bash
python mcp/test_mcp.py
```

### 4. SQL生成与校验测试

```bash
python -c "from codeact.generator import sql_generator; print('SQL生成器初始化成功')"
python -c "from codeact.validator import sql_validator; print('SQL校验器初始化成功')"
```

### 5. Agent Core测试

```bash
python -c "from agent.core import agent_core; result = agent_core.run('什么是固收类产品？'); print(result)"
```

---

## 常见问题

### 1. Elasticsearch连接问题

**问题**：无法连接到Elasticsearch

**解决方案**：
- 检查ES服务是否启动
- 检查`.env`文件中的ES配置是否正确
- 检查网络连接是否正常

### 2. 大模型API调用问题

**问题**：火山引擎API调用失败

**解决方案**：
- 检查API密钥是否正确
- 检查API配额是否充足
- 检查网络代理设置

### 3. 意图识别模型问题

**问题**：意图识别模型加载失败

**解决方案**：
- 系统会自动使用规则回退方案，不影响基本功能
- 检查训练数据是否充分
- 可以重新训练模型提升准确率

### 4. SQL生成失败

**问题**：生成的SQL不符合要求

**解决方案**：
- 检查表结构元数据是否正确
- 调整查询描述，使其更明确
- 检查是否违反了安全约束

### 5. SQL校验失败

**问题**：SQL校验不通过

**解决方案**：
- 检查是否使用了禁止的操作（增删改）
- 检查时间范围是否超过限制
- 检查是否有跨库JOIN或深层嵌套

### 6. 内存问题

**问题**：程序占用内存过多

**解决方案**：
- 减少返回的结果数量
- 调整向量检索的TopK值
- 定期清理缓存和内存

---

## 项目架构

```
ai_xiaoyi/
├── .env                  # 环境变量
├── main.py               # 程序入口
├── config.py             # 全局配置
├── agent/                # Agent核心层
│   ├── core.py           # Agent Core状态机
│   ├── memory.py         # 三级记忆系统
│   └── prompt.py         # 所有提示词模板
├── intent/               # 意图识别模块
│   └── recognizer.py     # 意图识别器
├── rag/                  # RAG知识库模块
│   └── retriever.py       # RAG检索器
├── mcp/                  # MCP协议与工具层
│   ├── protocol.py       # MCP协议定义
│   ├── skill_base.py     # Skill基类
│   ├── scheduler.py      # 工具调度器
│   └── skills/           # 具体Skill实现
│       ├── rag_skill.py
│       ├── echo_skill.py
│       └── sql_skill.py
├── codeact/              # CodeAct SQL模块
│   ├── generator.py      # SQL生成器
│   └── validator.py     # SQL校验器
└── utils/                # 工具函数
    ├── desensitize.py    # 数据脱敏
    ├── logger.py         # 日志工具
    └── token_counter.py  # Token计数
```

---

## 开发指南

### 添加新的Skill

1. 在`mcp/skills/`目录下创建新的Skill文件
2. 继承`SkillBase`类
3. 实现`execute()`、`get_description()`和`get_param_schema()`方法
4. 更新`mcp/skills/__init__.py`导出新的Skill

### 扩展记忆系统

1.根据需要调整短期/中期/长期记忆的容量
2. 实现自定义的存储后端
3. 添加记忆同步和清理机制

### 优化检索性能

1. 调整ES索引配置（分片、副本等）
2. 优化向量检索的TopK值
3. 实现结果缓存机制

### 增强SQL安全性

1. 添加更多的校验规则
2. 实现更复杂的AST分析
3. 添加SQL审计日志

---

## 安全说明

### SQL安全

- 禁止任何写操作（INSERT、UPDATE、DELETE等）
- 禁止跨数据库JOIN
- 限制嵌套查询深度不超过3层
- 限制时间范围不超过365天
- 限制结果集不超过1000行
- 禁止使用SELECT *

### 数据安全

- 实现数据脱敏（手机号、身份证、银行卡号等）
- 不在日志中输出敏感信息
- 使用最小权限原则配置数据库访问

### API安全

- 使用环境变量管理API密钥
- 不在代码中硬编码敏感信息
- 实现API调用限流和超时控制

---

## 性能优化建议

### 1. 向量检索优化

- 合理设置向量检索的TopK值
- 使用ES索引优化（分片、副本）
- 实现向量缓存

### 2. 大模型调用优化

- 实现请求批处理
- 使用提示词缓存
- 合理设置temperature参数

### 3. 记忆系统优化

- 实现记忆清理机制
- 使用高效的存储后端
- 限制历史记录的长度

### 4. 并发处理优化

- 使用异步执行提升吞吐量
- 实现请求队列和限流
- 避免资源竞争

---

## 许可证

本项目仅供内部学习和研究使用。使用本项目时请确保：

1. 遵守相关法律法规和公司政策
2. 不用于任何违法违规用途
3. 妥善处理用户数据，保护隐私信息
4. 获得必要的授权和许可

---

## 更新日志

### v1.0.0 (2026-04-30)

- 完成项目基础架构搭建
- 实现核心模块功能
- 编写完整的使用文档

---

## 联系方式

如有问题或建议，请联系项目维护团队。

---

**AI小益 - 让智慧运营更简单高效**
