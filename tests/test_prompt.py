"""
Prompt Manager 单元测试
测试提示词模板渲染、管理功能
"""

import pytest
from unittest.mock import Mock, patch

# 导入测试目标模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.prompt import PromptManager, TASK_DECOMPOSE_PROMPT, RESULT_AGGREGATE_PROMPT


@pytest.fixture
def prompt_manager():
    """创建提示词管理器实例"""
    return PromptManager()


# ==================== 测试提示词管理器基础功能 ====================

class TestPromptManagerBasics:
    """测试提示词管理器基础功能"""

    def test_initialization(self):
        """测试初始化"""
        manager = PromptManager()

        assert manager is not None
        assert isinstance(manager._prompts, dict)

    def test_has_prompt(self, prompt_manager):
        """测试检查提示词是否存在"""
        assert prompt_manager.has_prompt("task_decompose") is True
        assert prompt_manager.has_prompt("result_aggregate") is True
        assert prompt_manager.has_prompt("nonexistent") is False

    def test_list_prompts(self, prompt_manager):
        """测试列出所有提示词"""
        prompts = prompt_manager.list_prompts()

        assert isinstance(prompts, list)
        assert "task_decompose" in prompts
        assert "result_aggregate" in prompts
        assert len(prompts) == 2


# ==================== 测试提示词获取 ====================

class TestGetPrompt:
    """测试获取提示词模板"""

    def test_get_existing_prompt(self, prompt_manager):
        """测试获取存在的提示词"""
        prompt = prompt_manager.get_prompt("task_decompose")

        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "任务拆解" in prompt

    def test_get_nonexistent_prompt(self, prompt_manager):
        """测试获取不存在的提示词"""
        with pytest.raises(ValueError, match="提示词模板不存在"):
            prompt_manager.get_prompt("nonexistent_prompt")

    def test_get_prompt_no_formatting(self, prompt_manager):
        """测试获取不包含变量的提示词"""
        # 添加一个简单提示词
        prompt_manager.add_prompt("simple", "这是一个简单的提示词")

        prompt = prompt_manager.get_prompt("simple")

        assert prompt == "这是一个简单的提示词"


# ==================== 测试提示词模板渲染 ====================

class TestPromptTemplateRender:
    """测试提示词模板渲染"""

    def test_task_decompose_template_render(self, prompt_manager):
        """测试任务拆解提示词模板渲染"""
        prompt = prompt_manager.get_prompt(
            "task_decompose",
            user_query="什么是固收类产品？",
            intent_type="knowledge_query",
            intent_description="知识查询",
            confidence=0.8
        )

        assert "什么是固收类产品？" in prompt
        assert "knowledge_query" in prompt
        assert "知识查询" in prompt
        assert "0.8" in prompt

    def test_result_aggregate_template_render(self, prompt_manager):
        """测试结果聚合提示词模板渲染"""
        prompt = prompt_manager.get_prompt(
            "result_aggregate",
            user_query="查询产品收益",
            intent_type="data_query",
            intent_description="数据查询",
            tool_results="RAG结果: [...], SQL结果: [...]"
        )

        assert "查询产品收益" in prompt
        assert "data_query" in prompt
        assert "数据查询" in prompt
        assert "RAG结果" in prompt

    def test_template_render_with_complex_variables(self, prompt_manager):
        """测试复杂变量的模板渲染"""
        # 添加自定义提示词
        prompt_manager.add_prompt(
            "custom",
            "用户查询：{query}\n参数：{params}\n结果：{result}"
        )

        prompt = prompt_manager.get_prompt(
            "custom",
            query="测试查询",
            params={"key1": "value1", "key2": "value2"},
            result="执行成功"
        )

        assert "测试查询" in prompt
        assert "执行成功" in prompt

    def test_template_render_with_chinese_characters(self, prompt_manager):
        """测试中文字符的模板渲染"""
        prompt_manager.add_prompt(
            "chinese",
            "你好：{name}，欢迎使用：{system}"
        )

        prompt = prompt_manager.get_prompt(
            "chinese",
            name="张三",
            system="AI小益"
        )

        assert "张三" in prompt
        assert "AI小益" in prompt


# ==================== 测试模板渲染错误处理 ====================

class TestTemplateRenderErrors:
    """测试模板渲染错误处理"""

    def test_missing_template_variable(self, prompt_manager):
        """测试缺少模板变量"""
        prompt_manager.add_prompt(
            "test",
            "查询：{query}，参数：{params}"
        )

        with pytest.raises(ValueError, match="缺少模板变量"):
            prompt_manager.get_prompt("test", query="测试查询")
            # 缺少 params 变量

    def test_template_formatting_error(self, prompt_manager):
        """测试模板格式化错误"""
        prompt_manager.add_prompt(
            "test",
            "查询：{query}，数量：{.count}"
        )

        with pytest.raises(ValueError, match="提示词格式化失败"):
            prompt_manager.get_prompt("test", query="测试查询")

    def test_template_with_empty_variable_name(self, prompt_manager):
        """测试空变量名"""
        prompt_manager.add_prompt(
            "test",
            "内容：{}"
        )

        # Python 的 format 不允许空变量名
        with pytest.raises(ValueError):
            prompt_manager.get_prompt("test")


# ==================== 测试添加提示词 ====================

class TestAddPrompt:
    """测试添加提示词"""

    def test_add_new_prompt(self, prompt_manager):
        """测试添加新提示词"""
        initial_count = len(prompt_manager.list_prompts())

        prompt_manager.add_prompt("new_prompt", "这是新添加的提示词")

        assert len(prompt_manager.list_prompts()) == initial_count + 1
        assert prompt_manager.has_prompt("new_prompt") is True

    def test_add_prompt_overwrite_existing(self, prompt_manager):
        """测试覆盖已存在的提示词"""
        prompt_manager.add_prompt("test", "原始内容")
        assert "原始内容" in prompt_manager.get_prompt("test")

        # 覆盖
        prompt_manager.add_prompt("test", "新内容")
        assert "新内容" in prompt_manager.get_prompt("test")
        assert "原始内容" not in prompt_manager.get_prompt("test")

    def test_add_empty_prompt(self, prompt_manager):
        """测试添加空提示词"""
        prompt_manager.add_prompt("empty", "")

        assert prompt_manager.has_prompt("empty") is True
        assert prompt_manager.get_prompt("empty") == ""

    def test_add_multiline_prompt(self, prompt_manager):
        """测试添加多行提示词"""
        multiline = """这是第一行
这是第二行
这是第三行"""

        prompt_manager.add_prompt("multiline", multiline)

        prompt = prompt_manager.get_prompt("multiline")
        assert "这是第一行" in prompt
        assert "这是第二行" in prompt
        assert "这是第三行" in prompt

    def test_add_prompt_with_special_characters(self, prompt_manager):
        """测试添加特殊字符的提示词"""
        special = "特殊字符：@#$%^&*()_+-=[]{}|;':\",./<>?"

        prompt_manager.add_prompt("special", special)

        prompt = prompt_manager.get_prompt("special")
        assert "@" in prompt
        assert "#" in prompt


# ==================== 测试全局提示词常量 ====================

class TestGlobalPromptConstants:
    """测试全局提示词常量"""

    def test_task_decompose_prompt_exists(self):
        """测试任务拆解提示词常量存在"""
        assert TASK_DECOMPOSE_PROMPT is not None
        assert isinstance(TASK_DECOMPOSE_PROMPT, str)
        assert len(TASK_DECOMPOSE_PROMPT) > 0

    def test_task_decompose_prompt_content(self):
        """测试任务拆解提示词内容"""
        assert "任务规划助手" in TASK_DECOMPOSE_PROMPT
        assert "用户查询" in TASK_DECOMPOSE_PROMPT
        assert "意图识别结果" in TASK_DECOMPOSE_PROMPT
        assert "可用任务类型" in TASK_DECOMPOSE_PROMPT
        assert "rag_search" in TASK_DECOMPOSE_PROMPT
        assert "sql_execute" in TASK_DECOMPOSE_PROMPT

    def test_result_aggregate_prompt_exists(self):
        """测试结果聚合提示词常量存在"""
        assert RESULT_AGGREGATE_PROMPT is not None
        assert isinstance(RESULT_AGGREGATE_PROMPT, str)
        assert len(RESULT_AGGREGATE_PROMPT) > 0

    def test_result_aggregate_prompt_content(self):
        """测试结果聚合提示词内容"""
        assert "结果聚合助手" in RESULT_AGGREGATE_PROMPT
        assert "用户原始查询" in RESULT_AGGREGATE_PROMPT
        assert "意图类型" in RESULT_AGGREGATE_PROMPT
        assert "工具执行结果" in RESULT_AGGREGATE_PROMPT


# ==================== 测试完整提示词渲染流程 ====================

class TestCompletePromptFlow:
    """测试完整提示词渲染流程"""

    def test_complete_task_decompose_flow(self, prompt_manager):
        """测试完整的任务拆解提示词渲染流程"""
        # 模拟真实的参数
        render_params = {
            "user_query": "近30天固收类产品总收益",
            "intent_type": "data_query",
            "intent_description": "数据查询",
            "confidence": 0.85
        }

        prompt = prompt_manager.get_prompt("task_decompose", **render_params)

        # 验证所有变量都被替换
        assert "{user_query}" not in prompt
        assert "{intent_type}" not in prompt
        assert "{intent_description}" not in prompt
        assert "{confidence}" not in prompt

        # 验证实际值存在
        assert "近30天固收类产品总收益" in prompt
        assert "data_query" in prompt
        assert "数据查询" in prompt
        assert "0.85" in prompt

    def test_complete_result_aggregate_flow(self, prompt_manager):
        """测试完整的结果聚合提示词渲染流程"""
        # 模拟真实的参数
        render_params = {
            "user_query": "查询产品收益",
            "intent_type": "data_query",
            "intent_description": "数据查询",
            "tool_results": """
            任务1 (rag_search): 成固收类产品是固定收益类理财产品
            任务2 (sql_execute): 执行成功，返回 5 条记录
            """
        }

        prompt = prompt_manager.get_prompt("result_aggregate", **render_params)

        # 验证所有变量都被替换
        assert "{user_query}" not in prompt
        assert "{intent_type}" not in prompt
        assert "{intent_description}" not in prompt
        assert "{tool_results}" not in prompt

        # 验证实际值存在
        assert "查询产品收益" in prompt
        assert "data_query" in prompt
        assert "数据查询" in prompt
        assert "rag_search" in prompt

    def test_render_with_nested_dict(self, prompt_manager):
        """测试嵌套字典的渲染"""
        prompt_manager.add_prompt(
            "nested",
            "查询：{query}，参数：params[key1]={params[key1]}, params[key2]={params[key2]}"
        )

        prompt = prompt_manager.get_prompt(
            "nested",
            query="测试",
            params={"key1": "value1", "key2": "value2"}
        )

        assert "测试" in prompt
        assert "value1" in prompt
        assert "value2" in prompt


# ==================== 测试边缘情况 ====================

class TestEdgeCases:
    """测试边缘情况"""

    def test_render_with_very_long_string(self, prompt_manager):
        """测试超长字符串渲染"""
        long_string = "A" * 10000

        prompt_manager.add_prompt("long", "内容：{content}")
        prompt = prompt_manager.get_prompt("long", content=long_string)

        assert len(prompt) > 10000
        assert long_string in prompt

    def test_render_with_unicode(self, prompt_manager):
        """测试 Unicode 字符渲染"""
        unicode_text = "你好世界 🌍 🎉 🚀 ✓ ✗"

        prompt_manager.add_prompt("unicode", "文本：{text}")
        prompt = prompt_manager.get_prompt("unicode", text=unicode_text)

        assert "你好世界" in prompt
        assert "🌍" in prompt

    def test_render_with_numbers(self, prompt_manager):
        """测试数字类型渲染"""
        prompt_manager.add_prompt(
            "numbers",
            "整数：{int_val}，浮点数：{float_val}，科学计数：{sci_val}"
        )

        prompt = prompt_manager.get_prompt(
            "numbers",
            int_val=42,
            float_val=3.14159,
            sci_val=1.23e-5
        )

        assert "42" in prompt
        assert "3.14159" in prompt

    def test_render_with_none_value(self, prompt_manager):
        """测试 None 值渲染"""
        prompt_manager.add_prompt("none", "值：{value}")

        prompt = prompt_manager.get_prompt("none", value=None)

        assert "None" in prompt

    def test_render_with_boolean(self, prompt_manager):
        """测试布尔值渲染"""
        prompt_manager.add_prompt("bool", "成功：{success}, 失败：{failed}")

        prompt = prompt_manager.get_prompt("bool", success=True, failed=False)

        assert "True" in prompt
        assert "False" in prompt


# ==================== 测试提示词管理器并发安全 ====================

class TestPromptManagerConcurrency:
    """测试提示词管理器并发访问"""

    def test_multiple_get_requests(self, prompt_manager):
        """测试多次获取请求"""
        prompt_manager.add_prompt("test", "测试内容：{value}")

        # 多次获取
        for i in range(10):
            prompt = prompt_manager.get_prompt("test", value=i)
            assert str(i) in prompt

    def test_interleaved_add_and_get(self, prompt_manager):
        """测试交替添加和获取"""
        # 添加多个提示词
        for i in range(5):
            prompt_manager.add_prompt(f"prompt_{i}", f"内容{i}")

        # 验证都能获取
        for i in range(5):
            assert prompt_manager.has_prompt(f"prompt_{i}") is True
            prompt = prompt_manager.get_prompt(f"prompt_{i}")
            assert f"内容{i}" in prompt


# ==================== 测试提示词命名约定 ====================

class TestPromptNamingConventions:
    """测试提示词命名约定"""

    def test_prompt_name_with_underscores(self, prompt_manager):
        """测试带下划线的提示词名称"""
        prompt_manager.add_prompt("test_prompt_name", "测试内容")

        assert prompt_manager.has_prompt("test_prompt_name") is True

    def test_prompt_name_with_numbers(self, prompt_manager):
        """测试带数字的提示词名称"""
        prompt_manager.add_prompt("prompt123", "测试内容")

        assert prompt_manager.has_prompt("prompt123") is True

    def test_prompt_name_case_sensitivity(self, prompt_manager):
        """测试提示词名称大小写敏感性"""
        prompt_manager.add_prompt("TestPrompt", "大写内容")

        assert prompt_manager.has_prompt("TestPrompt") is True
        assert prompt_manager.has_prompt("testprompt") is False


# ==================== 测试提示词模板语法 ====================

class TestTemplateSyntax:
    """测试模板语法"""

    def test_template_with_format_specifier(self, prompt_manager):
        """测试带格式说明符的模板"""
        prompt_manager.add_prompt(
            "format",
            "数字：{num:.2f}，字符串：{str:>10}"
        )

        prompt = prompt_manager.get_prompt(
            "format",
            num=3.14159,
            str="test"
        )

        assert "3.14" in prompt

    def test_template_with_indexed_placeholder(self, prompt_manager):
        """测试带索引占位符的模板"""
        prompt_manager.add_prompt(
            "indexed",
            "第一：{0}，第二：{1}，第一重复：{0}"
        )

        prompt = prompt_manager.get_prompt(
            "indexed",
            "值1",
            "值2"
        )

        assert "值1" in prompt
        assert "值2" in prompt
        assert prompt.count("值1") == 2
