"""
Agent Core 单元测试
测试任务拆解、工具执行、结果聚合、完整流程
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from typing import Dict, Any

# 导入测试目标模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_llm():
    """Mock LLM 客户端"""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="模拟的LLM响应")
    return mock


@pytest.fixture
def mock_intent_recognizer():
    """Mock 意图识别器"""
    from intent.recognizer import IntentResult
    mock = MagicMock()

    # 设置返回值
    mock.predict.return_value = IntentResult(
        intent="knowledge_query",
        confidence=0.8,
        clarification=None
    )
    return mock


@pytest.fixture
def mock_skill_scheduler():
    """Mock 技能调度器"""
    from mcp.protocol import MCPResponse
    mock = MagicMock()

    # 设置返回成功响应
    mock.execute.return_value = MCPResponse.success_response(
        data={"message": "测试成功", "results": [{"content": "测试内容"}]},
        trace_id="test_trace_id",
        execution_time=100.0
    )
    return mock


@pytest.fixture
def mock_memory_system():
    """Mock 记忆系统"""
    mock = MagicMock()
    return mock


@pytest.fixture
def agent_core_with_mocks(mock_llm, mock_intent_recognizer, mock_skill_scheduler, mock_memory_system):
    """创建带有完整 mock 的 AgentCore 实例"""
    from agent.core import AgentCore, AgentState, IntentResult, AgentMemory

    with patch('agent.core.config') as mock_config, \
         patch('agent.core.intent_recognizer', mock_intent_recognizer), \
         patch('agent.core.skill_scheduler', mock_skill_scheduler), \
         patch('agent.core.memory_system', mock_memory_system):

        # Mock config
        mock_config.llm_client = mock_llm
        mock_config.es_client = None

        # 创建 AgentCore 实例
        agent = AgentCore()
        agent._intent_recognizer = mock_intent_recognizer
        agent._skill_scheduler = mock_skill_scheduler
        agent._memory_system = mock_memory_system
        agent._initialized = True

        return agent


# ==================== 测试任务拆解 ====================

class TestTaskDecompose:
    """测试任务拆解功能"""

    def test_task_decompose_knowledge_query(self, agent_core_with_mocks):
        """测试知识查询意图的任务拆解"""
        from intent.recognizer import IntentResult

        agent = agent_core_with_mocks
        query = "什么是固收类产品？"
        intent = IntentResult(
            intent="knowledge_query",
            confidence=0.8,
            clarification=None
        )

        tasks = agent._decompose_tasks(query, intent)

        assert len(tasks) == 1
        assert tasks[0]["task_type"] == "rag_search"
        assert "query" in tasks[0]["params"]
        assert tasks[0]["params"]["query"] == query

    def test_task_decompose_data_query(self, agent_core_with_mocks):
        """测试数据查询意图的任务拆解"""
        from intent.recognizer import IntentResult

        agent = agent_core_with_mocks
        query = "查询近30天产品收益"
        intent = IntentResult(
            intent="data_query",
            confidence=0.8,
            clarification=None
        )

        tasks = agent._decompose_tasks(query, intent)

        assert len(tasks) == 1
        assert tasks[0]["task_type"] == "sql_execute"
        assert "query" in tasks[0]["params"]
        assert tasks[0]["params"]["query"] == query

    def test_task_decompose_task_operation(self, agent_core_with_mocks):
        """测试任务操作意图的任务拆解"""
        from intent.recognizer import IntentResult

        agent = agent_core_with_mocks
        query = "执行操作：创建订单"
        intent = IntentResult(
            intent="task_operation",
            confidence=0.8,
            clarification=None
        )

        tasks = agent._decompose_tasks(query, intent)

        assert len(tasks) == 1
        assert tasks[0]["task_type"] == "echo"
        assert "message" in tasks[0]["params"]

    def test_task_decompose_unknown_intent(self, agent_core_with_mocks):
        """测试未知意图的任务拆解"""
        from intent.recognizer import IntentResult

        agent = agent_core_with_mocks
        query = "未知请求"
        intent = IntentResult(
            intent="unknown",
            confidence=0.5,
            clarification=None
        )

        tasks = agent._decompose_tasks(query, intent)

        assert len(tasks) == 0

    def test_task_decompose_node(self, agent_core_with_mocks):
        """测试任务拆解节点"""
        from intent.recognizer import IntentResult
        from agent.core import AgentState

        agent = agent_core_with_mocks

        state: AgentState = {
            "user_query": "什么是固收类产品？",
            "intent": IntentResult(
                intent="knowledge_query",
                confidence=0.8,
                clarification=None
            ),
            "task_list": [],
            "current_task_idx": 0,
            "tool_results": [],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": None
        }

        new_state = agent._task_decompose_node(state)

        assert len(new_state["task_list"]) == 1
        assert new_state["current_task_idx"] == 0
        assert new_state["step_count"] == 1


# ==================== 测试工具执行 ====================

class TestToolExecute:
    """测试工具执行功能"""

    def test_execute_rag_search_success(self, agent_core_with_mocks, mock_skill_scheduler):
        """测试 RAG 搜索成功执行"""
        from mcp.protocol import MCPResponse

        agent = agent_core_with_mocks

        task = {
            "task_id": "task_1",
            "task_type": "rag_search",
            "description": "检索知识库",
            "params": {
                "query": "固收类产品",
                "business_domain": "finance"
            }
        }

        # Mock 返回值
        mock_skill_scheduler.execute.return_value = MCPResponse.success_response(
            data={"results": [{"content": "固收类产品是固定收益类产品"}]},
            trace_id="task_1"
        )

        result = agent._execute_rag_search(task)

        assert result["success"] is True
        assert "results" in result["data"]

    def test_execute_sql_task_success(self, agent_core_with_mocks, mock_skill_scheduler):
        """测试 SQL 任务成功执行"""
        from mcp.protocol import MCPResponse

        agent = agent_core_with_mocks

        task = {
            "task_id": "task_1",
            "task_type": "sql_execute",
            "description": "执行SQL查询",
            "params": {
                "query": "查询产品收益",
                "table_name": "product_info"
            }
        }

        # Mock 返回值
        mock_skill_scheduler.execute.return_value = MCPResponse.success_response(
            data={"sql": "SELECT * FROM product_info", "rows": []},
            trace_id="task_1"
        )

        result = agent._execute_sql_task(task)

        assert result["success"] is True
        assert "sql" in result["data"]

    def test_execute_echo_task_success(self, agent_core_with_mocks, mock_skill_scheduler):
        """测试 Echo 任务成功执行"""
        from mcp.protocol import MCPResponse

        agent = agent_core_with_mocks

        task = {
            "task_id": "task_1",
            "task_type": "echo",
            "description": "Echo测试",
            "params": {
                "message": "测试消息"
            }
        }

        # Mock 返回值
        mock_skill_scheduler.execute.return_value = MCPResponse.success_response(
            data={"message": "测试消息"},
            trace_id="task_1"
        )

        result = agent._execute_echo_task(task)

        assert result["success"] is True
        assert result["data"]["message"] == "测试消息"

    def test_execute_unknown_task_type(self, agent_core_with_mocks):
        """测试未知任务类型"""
        from agent.core import AgentState

        agent = agent_core_with_mocks

        state: AgentState = {
            "user_query": "测试查询",
            "intent": None,
            "task_list": [{
                "task_id": "task_1",
                "task_type": "unknown_task",
                "description": "未知任务",
                "params": {}
            }],
            "current_task_idx": 0,
            "tool_results": [],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": None
        }

        new_state = agent._tool_execute_node(state)

        assert len(new_state["tool_results"]) == 1
        assert new_state["tool_results"][0]["result"]["success"] is False
        assert "未知任务类型" in new_state["tool_results"][0]["result"]["error"]

    def test_tool_execute_with_retry(self, agent_core_with_mocks, mock_skill_scheduler):
        """测试工具执行带重试机制"""
        from mcp.protocol import MCPResponse
        from agent.core import AgentState

        agent = agent_core_with_mocks

        # 第一次失败，第二次成功
        mock_skill_scheduler.execute.side_effect = [
            MCPResponse.error_response(
                error="连接失败",
                trace_id="task_1"
            ),
            MCPResponse.success_response(
                data={"message": "成功"},
                trace_id="task_1"
            )
        ]

        state: AgentState = {
            "user_query": "测试查询",
            "intent": None,
            "task_list": [{
                "task_id": "task_1",
                "task_type": "echo",
                "description": "测试",
                "params": {"message": "测试"}
            }],
            "current_task_idx": 0,
            "tool_results": [],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": None
        }

        # 第一次执行（失败）
        agent._tool_execute_node(state)
        assert state["error"] is not None

        # 第二次执行（成功）
        state["error"] = None
        agent._tool_execute_node(state)
        assert state["error"] is None


# ==================== 测试结果聚合 ====================

class TestResultAggregate:
    """测试结果聚合功能"""

    def test_result_aggregate_with_successful_results(self, agent_core_with_mocks):
        """测试有成功结果的情况"""
        from agent.core import AgentState

        agent = agent_core_with_mocks

        state: AgentState = {
            "user_query": "什么是固收类产品？",
            "intent": None,
            "task_list": [],
            "current_task_idx": 0,
            "tool_results": [
                {
                    "task_id": "task_1",
                    "task_type": "rag_search",
                    "result": {
                        "success": True,
                        "data": {
                            "results": [
                                {"content": "固收类产品是固定收益类理财产品"},
                                {"content": "具有较低风险"}
                            ]
                        }
                    }
                }
            ],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": None
        }

        new_state = agent._result_aggregate_node(state)

        assert new_state["final_answer"] is not None
        assert "固收类产品" in new_state["final_answer"]

    def test_result_aggregate_with_failed_results(self, agent_core_with_mocks):
        """测试只有失败结果的情况"""
        from agent.core import AgentState

        agent = agent_core_with_mocks

        state: AgentState = {
            "user_query": "测试查询",
            "intent": None,
            "task_list": [],
            "current_task_idx": 0,
            "tool_results": [
                {
                    "task_id": "task_1",
                    "task_type": "rag_search",
                    "result": {
                        "success": False,
                        "error": "知识库连接失败"
                    }
                }
            ],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": None
        }

        new_state = agent._result_aggregate_node(state)

        assert new_state["final_answer"] is not None
        assert "抱歉" in new_state["final_answer"]

    def test_result_aggregate_with_empty_results(self, agent_core_with_mocks):
        """测试空结果列表"""
        from agent.core import AgentState

        agent = agent_core_with_mocks

        state: AgentState = {
            "user_query": "测试查询",
            "intent": None,
            "task_list": [],
            "current_task_idx": 0,
            "tool_results": [],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": None
        }

        new_state = agent._result_aggregate_node(state)

        assert new_state["final_answer"] is not None

    def test_result_aggregate_mixed_results(self, agent_core_with_mocks):
        """测试混合成功和失败结果"""
        from agent.core import AgentState

        agent = agent_core_with_mocks

        state: AgentState = {
            "user_query": "测试查询",
            "intent": None,
            "task_list": [],
            "current_task_idx": 0,
            "tool_results": [
                {
                    "task_id": "task_1",
                    "task_type": "rag_search",
                    "result": {
                        "success": True,
                        "data": {
                            "results": [{"content": "查询成功"}]
                        }
                    }
                },
                {
                    "task_id": "task_2",
                    "task_type": "sql_execute",
                    "result": {
                        "success": False,
                        "error": "SQL执行失败"
                    }
                }
            ],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": None
        }

        new_state = agent._result_aggregate_node(state)

        assert new_state["final_answer"] is not None
        assert "查询成功" in new_state["final_answer"]
        assert "失败" in new_state["final_answer"]


# ==================== 测试完整流程 ====================

class TestAgentWorkflow:
    """测试 Agent 完整工作流"""

    def test_full_workflow_knowledge_query(self, agent_core_with_mocks, mock_skill_scheduler):
        """测试知识查询完整流程"""
        from mcp.protocol import MCPResponse
        from intent.recognizer import IntentResult

        agent = agent_core_with_mocks

        # Mock 意图识别
        agent._intent_recognizer.predict.return_value = IntentResult(
            intent="knowledge_query",
            confidence=0.8,
            clarification=None
        )

        # Mock 技能执行
        mock_skill_scheduler.execute.return_value = MCPResponse.success_response(
            data={"results": [{"content": "固收类产品是固定收益类产品"}]},
            trace_id="test_trace"
        )

        result = agent.run("什么是固收类产品？")

        assert result["success"] is True
        assert result["final_answer"] is not None
        assert len(result["task_list"]) > 0
        assert len(result["tool_results"]) > 0

    def test_full_workflow_data_query(self, agent_core_with_mocks, mock_skill_scheduler):
        """测试数据查询完整流程"""
        from mcp.protocol import MCPResponse
        from intent.recognizer import IntentResult

        agent = agent_core_with_mocks

        # Mock 意图识别
        agent._intent_recognizer.predict.return_value = IntentResult(
            intent="data_query",
            confidence=0.8,
            clarification=None
        )

        # Mock 技能执行
        mock_skill_scheduler.execute.return_value = MCPResponse.success_response(
            data={"sql": "SELECT * FROM product_info", "rows": []},
            trace_id="test_trace"
        )

        result = agent.run("查询产品收益")

        assert result["success"] is True
        assert result["final_answer"] is not None

    def test_stream_workflow(self, agent_core_with_mocks, mock_skill_scheduler):
        """测试流式执行流程"""
        from mcp.protocol import MCPResponse
        from intent.recognizer import IntentResult

        agent = agent_core_with_mocks

        # Mock 意图识别
        agent._intent_recognizer.predict.return_value = IntentResult(
            intent="knowledge_query",
            confidence=0.8,
            clarification=None
        )

        # Mock 技能执行
        mock_skill_scheduler.execute.return_value = MCPResponse.success_response(
            data={"results": [{"content": "测试内容"}]},
            trace_id="test_trace"
        )

        events = list(agent.stream_run("测试查询"))

        assert len(events) > 0

    def test_workflow_with_error(self, agent_core_with_mocks):
        """测试带错误的流程"""
        from agent.core import AgentState
        from intent.recognizer import IntentResult

        agent = agent_core_with_mocks

        # Mock 意图识别
        agent._intent_recognizer.predict.return_value = IntentResult(
            intent="unknown",
            confidence=0.3,
            clarification=None
        )

        result = agent.run("未知请求")

        assert "final_answer" in result

    def test_workflow_max_steps_limit(self, agent_core_with_mocks):
        """测试最大步数限制"""
        from agent.core import AgentState

        agent = agent_core_with_mocks

        state: AgentState = {
            "user_query": "测试",
            "intent": None,
            "task_list": [],
            "current_task_idx": 0,
            "tool_results": [],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 10,  # 超过最大步数
            "error": None
        }

        next_step = agent._check_execution_status(state)

        assert next_step == "all_tasks_done"


# ==================== 测试错误处理 ====================

class TestErrorHandling:
    """测试错误处理功能"""

    def test_friendly_error_not_found(self, agent_core_with_mocks):
        """测试 not found 错误的友好提示"""
        agent = agent_core_with_mocks
        error_msg = agent._generate_friendly_error("resource not found")
        assert "找不到" in error_msg

    def test_friendly_error_timeout(self, agent_core_with_mocks):
        """测试 timeout 错误的友好提示"""
        agent = agent_core_with_mocks
        error_msg = agent._generate_friendly_error("request timeout")
        assert "超时" in error_msg

    def test_friendly_error_permission(self, agent_core_with_mocks):
        """测试 permission 错误的友好提示"""
        agent = agent_core_with_mocks
        error_msg = agent._generate_friendly_error("permission denied")
        assert "权限" in error_msg

    def test_friendly_error_validation(self, agent_core_with_mocks):
        """与其他 validation 错误的友好提示"""
        agent = agent_core_with_mocks
        error_msg = agent._generate_friendly_error("validation error")
        assert "格式不正确" in error_msg

    def test_friendly_error_unknown(self, agent_core_with_mocks):
        """测试未知错误的友好提示"""
        agent = agent_core_with_mocks
        error_msg = agent._generate_friendly_error("unknown error")
        assert "抱歉" in error_msg

    def test_error_handle_node_retry(self, agent_core_with_mocks):
        """测试错误处理节点重试"""
        from agent.core import AgentState

        agent = agent_core_with_mocks

        state: AgentState = {
            "user_query": "测试",
            "intent": None,
            "task_list": [],
            "current_task_idx": 0,
            "tool_results": [],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": "测试错误"
        }

        new_state = agent._error_handle_node(state)

        assert new_state["retry_count"] == 1
        assert new_state["error"] is None  # 错误被清除

    def test_error_handle_node_max_retry(self, agent_core_with_mocks):
        """测试超过最大重试次数"""
        from agent.core import AgentState, MAX_RETRY_COUNT

        agent = agent_core_with_mocks

        state: AgentState = {
            "user_query": "测试",
            "intent": None,
            "task_list": [],
            "current_task_idx": 0,
            "tool_results": [],
            "final_answer": None,
            "memory": None,
            "retry_count": MAX_RETRY_COUNT,  # 已达到最大重试次数
            "step_count": 0,
            "error": "测试错误"
        }

        new_state = agent._error_handle_node(state)

        assert new_state["final_answer"] is not None
        assert new_state["error"] is None


# ==================== 测试初始化 ====================

class TestInitialization:
    """测试 AgentCore 初始化"""

    def test_agent_initialization(self, agent_core_with_mocks):
        """测试 AgentCore 初始化"""
        agent = agent_core_with_mocks
        assert agent._initialized is True
        assert agent._graph is not None

    def test_build_graph(self, agent_core_with_mocks):
        """测试状态图构建"""
        agent = agent_core_with_mocks
        assert agent._graph is not None

    def test_has_tasks_condition(self, agent_core_with_mocks):
        """测试任务检查条件"""
        from agent.core import AgentState

        agent = agent_core_with_mocks

        state_with_tasks: AgentState = {
            "user_query": "测试",
            "intent": None,
            "task_list": [{"task_id": "task_1"}],
            "current_task_idx": 0,
            "tool_results": [],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": None
        }

        assert agent._has_tasks(state_with_tasks) == "has_tasks"

        state_no_tasks: AgentState = {
            "user_query": "测试",
            "intent": None,
            "task_list": [],
            "current_task_idx": 0,
            "tool_results": [],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": None
        }

        assert agent._has_tasks(state_no_tasks) == "no_tasks"


# ==================== 测试意图识别 ====================

class TestIntentRecognize:
    """测试意图识别功能"""

    def test_intent_recognize_node(self, agent_core_with_mocks):
        """测试意图识别节点"""
        from intent.recognizer import IntentResult
        from agent.core import AgentState

        agent = agent_core_with_mocks

        state: AgentState = {
            "user_query": "什么是固收类产品？",
            "intent": None,
            "task_list": [],
            "current_task_idx": 0,
            "tool_results": [],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": None
        }

        new_state = agent._intent_recognize_node(state)

        assert new_state["intent"] is not None
        assert new_state["step_count"] == 1

    def test_mock_intent_recognize(self, agent_core_with_mocks):
        """测试模拟意图识别"""
        agent = agent_core_with_mocks

        # 测试知识查询关键词
        result = agent._mock_intent_recognize("什么是固收类产品？")
        assert result.intent == "knowledge_query"

        # 测试数据查询关键词
        result = agent._mock_intent_recognize("查询产品收益")
        assert result.intent == "data_query"

        # 测试任务操作关键词
        result = agent._mock_intent_recognize("执行创建订单")
        assert result.intent == "task_operation" or result.intent == "knowledge_query"


# ==================== 测试运行时异常 ====================

class TestRuntimeExceptions:
    """测试运行时异常处理"""

    def test_run_without_initialization(self):
        """测试未初始化时运行"""
        from agent.core import AgentCore

        agent = AgentCore()
        agent._initialized = False

        with pytest.raises(RuntimeError, match="Agent Core未初始化"):
            agent.run("测试查询")

    def test_stream_run_without_initialization(self):
        """测试未初始化时流式运行"""
        from agent.core import AgentCore

        agent = AgentCore()
        agent._initialized = False

        with pytest.raises(RuntimeError, match="Agent Core未初始化"):
            list(agent.stream_run("测试查询"))

    def test_intent_recognize_with_exception(self, agent_core_with_mocks):
        """测试意图识别异常处理"""
        from agent.core import AgentState

        agent = agent_core_with_mocks
        agent._intent_recognizer.predict.side_effect = Exception("识别失败")

        state: AgentState = {
            "user_query": "测试",
            "intent": None,
            "task_list": [],
            "current_task_idx": 0,
            "tool_results": [],
            "final_answer": None,
            "memory": None,
            "retry_count": 0,
            "step_count": 0,
            "error": None
        }

        new_state = agent._intent_recognize_node(state)

        assert new_state["error"] is not None
        assert "意图识别失败" in new_state["error"]
