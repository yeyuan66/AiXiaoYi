你是一名资深agent开发专家，现在需要在当前目录下开发一款名为“AI小益”的智能化运营助手。项目概述：
1、服务对象：招商银行收益升级中台系统的业务人员
2、项目技术栈：langchain做agent搭建框架、elasticSearch做向量数据库、scikit-learn做意图识别等
3、功能支持：意图识别、agent core的规划，任务拆解，结果聚合，调用mcp工具，mcp协议格式，调用rag知识库完成理财产品业务知识检索召回，提示词工程，三级记忆系统，调用各种skill包括生成动态SQL调用安全沙箱并完成校验，结果流式输出


请基于以下要求，完成AI小益项目的基础配置模块：
1. 读取.env文件中的所有配置，封装成Config单例类
2. 初始化火山引擎GLM4.7大模型客户端（使用langchain-openai的ChatOpenAI）
3. 初始化Elasticsearch客户端，连接到配置中的ES实例
4. 定义全局常量：最大重试次数、最大执行步数、输入Token上限、输出预留Token
5. 实现基础工具函数：Token计数函数（使用tiktoken，适配GLM模型）、数据脱敏函数（手机号、身份证、银行卡号）
6. 所有代码必须符合Python3.10+规范，使用类型注解
7. 输出config.py和utils/token_counter.py、utils/desensitize.py三个文件的完整代码

并生成以下的目录结构：
ai_xiaoyi/
├── .env                  # 环境变量
├── main.py               # 程序入口
├── config.py             # 全局配置
├── agent/                # Agent核心层
│   ├── core.py           # Agent Core状态机
│   ├── memory.py         # 三级记忆系统
│   └── prompt.py         # 所有提示词模板
├── intent/               # 意图识别模块
│   └── recognizer.py
├── rag/                  # RAG知识库模块
│   └── retriever.py
├── mcp/                  # MCP协议与工具层
│   ├── protocol.py       # MCP协议定义
│   ├── skill_base.py     # Skill基类
│   ├── skills/           # 具体Skill实现
│   │   ├── rag_skill.py
│   │   └── sql_skill.py
│   └── scheduler.py      # 工具调度器
├── codeact/              # CodeAct SQL模块
│   ├── generator.py      # SQL生成
│   └── validator.py      # SQL校验
└── utils/                # 工具函数
    ├── desensitize.py    # 数据脱敏
    └── logger.py         # 日志工具

技术栈要求：
- 大模型：langchain-openai调用火山引擎GLM4.7
- 向量库：elasticsearch==7.17.14
- 配置管理：python-dotenv