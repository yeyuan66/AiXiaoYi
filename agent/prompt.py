"""
提示词模板模块
存储所有Agent相关的提示词模板
"""

# 任务拆解提示词
TASK_DECOMPOSE_PROMPT = """你是一个任务规划助手，负责将用户的自然语言查询拆解为可执行的原子任务。

## 用户查询
{user_query}

## 意图识别结果
- 意图类型: {intent_type}
- 意图描述: {intent_description}
- 置信度: {confidence}

## 可用任务类型
1. **rag_search**: RAG知识库检索
   - 描述: 从知识库中检索相关信息
   - 适用场景: 产品知识查询、业务规则询问等

2. **sql_execute**: SQL生成和执行
   - 描述: 生成并执行SQL查询
   - 适用场景: 数据统计、业绩查询、报表生成等

3. **echo**: 测试回显
   - 描述: 简单返回输入内容
   - 适用场景: 测试、调试

## 要求
1. 分析用户查询，识别需要执行的任务
2. 如果查询复杂，拆解为多个任务
3. 每个任务必须明确指定任务类型和参数
4. 按照任务执行顺序排序（有依赖关系时）
5. 输出格式为JSON数组，每个任务包含：
   - task_id: 任务唯一标识
   - task_type: 任务类型
   - description: 任务描述
   - params: 任务参数
   - depends_on: 依赖的其他任务ID（可选）

示例：
查询"近30天固收类产品总收益" ->
```json
[
  {
    "task_id": "task_1",
    "task_type": "rag_search",
    "description": "检索固收类产品相关知识",
    "params": {
      "query": "固收类产品",
      "business_domain": "finance"
    }
  },
  {
    "task_id": "task_2",
    "task_type": "sql_execute",
    "description": "查询近30天产品收益数据",
    "params": {
      "query": "近30天产品收益",
      "table_name": "product_info"
    },
    "depends_on": ["task_1"]
  }
]
```

只返回JSON格式的任务列表，不要包含其他内容。
"""


# 结果聚合提示词
RESULT_AGGREGATE_PROMPT = """你是一个结果聚合助手，负责将多个工具执行结果整合为自然语言回答。

## 用户原始查询
{user_query}

## 意图类型
- 意图类型: {intent_type}
- 意图描述: {intent_description}

## 工具执行结果
{tool_results}

## 要求
1. 分析所有工具执行结果
2. 根据用户查询的意图，生成清晰、准确的自然语言回答
3. 如果工具执行失败，提供友好的错误提示
4. 回答应该：
   - 直接回答用户的问题
   - 引用相关的工具结果
   - 提供必要的数据或信息
   - 语言简洁、专业

## 回答风格指南
- 使用简洁、专业的语言
- 避免过于技术化的术语
- 提供具体、有价值的信息
- 如果数据很多，进行合理的总结

只返回自然语言回答，不要包含其他格式或解释。
"""


class PromptManager:
    """提示词管理器"""

    def __init__(self):
        """初始化提示词管理器"""
        self._prompts = {
            "task_decompose": TASK_DECOMPOSE_PROMPT,
            "result_aggregate": RESULT_AGGREGATE_PROMPT
        }

    def get_prompt(self, prompt_name: str, **kwargs) -> str:
        """
        获取提示词模板

        Args:
            prompt_name: 提示词名称
            **kwargs: 模板变量

        Returns:
            格式化后的提示词
        """
        template = self._prompts.get(prompt_name)

        if template is None:
            raise ValueError(f"提示词模板不存在: {prompt_name}")

        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"缺少模板变量: {e}")
        except Exception as e:
            raise ValueError(f"提示词格式化失败: {e}")

    def add_prompt(self, name: str, template: str) -> None:
        """
        添加提示词模板

        Args:
            name: 提示词名称
            template: 提示词模板
        """
        self._prompts[name] = template

    def list_prompts(self) -> list[str]:
        """
        列出所有提示词名称

        Returns:
            提示词名称列表
        """
        return list(self._prompts.keys())

    def has_prompt(self, name: str) -> bool:
        """
        检查提示词是否存在

        Args:
            name: 提示词名称

        Returns:
            是否存在
        """
        return name in self._prompts


# 创建全局提示词管理器实例
prompt_manager = PromptManager()
