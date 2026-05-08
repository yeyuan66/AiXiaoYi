"""
Agent Core状态机模块
基于LangGraph实现任务拆解、多步规划、结果聚合
"""

from typing import TypedDict, List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from mcp.scheduler import skill_scheduler, SkillScheduler
from mcp.protocol import MCPRequest, MCPResponse

if TYPE_CHECKING:
    from intent.recognizer import IntentRecognizer
    from .memory import MemorySystem

try:
    from config import config, MAX_EXECUTE_STEPS, MAX_RETRY_COUNT
    from utils import default_logger
    from intent.recognizer import IntentRecognizer, IntentResult
    from .memory import MemorySystem, AgentMemory
    from .prompt import (
        TASK_DECOMPOSE_PROMPT,
        RESULT_AGGREGATE_PROMPT
    )
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    default_logger = logging.getLogger('agent_core')
    MAX_EXECUTE_STEPS = 3
    MAX_RETRY_COUNT = 2

    # 定义模拟类以避免导入错误
    @dataclass
    class IntentResult:
        intent: str = "unknown"
        confidence: float = 0.0
        metadata: Dict[str, Any] = None

    @dataclass
    class AgentMemory:
        session_id: str = ""
        user_id: Optional[str] = None
        current_query: str = ""
        context: Optional[str] = None
        preferences: Dict[str, Any] = None
        conversation_history: List[Dict[str, Any]] = None


# 定义AgentState状态类
class AgentState(TypedDict):
    """Agent状态定义"""
    user_query: str                    # 用户原始查询
    intent: Optional[IntentResult]      # 意图识别结果
    task_list: List[Dict[str, Any]]   # 任务列表
    current_task_idx: int               # 当前任务索引
    tool_results: List[Dict[str, Any]] # 工具执行结果
    final_answer: Optional[str]          # 最终答案
    memory: Optional[AgentMemory]        # 记忆系统
    retry_count: int                    # 重试次数
    step_count: int                     # 执行步数
    error: Optional[str]                 # 错误信息


class AgentCore:
    """
    Agent Core核心类
    基于LangGraph实现任务拆解、多步规划、结果聚合
    """

    def __init__(self):
        """初始化Agent Core"""
        self._graph = None
        self._llm = None
        self._intent_recognizer: Optional[IntentRecognizer] = None
        self._skill_scheduler: Optional[SkillScheduler] = None
        self._memory_system: Optional[MemorySystem] = None
        self._initialized = False

        self._initialize()

    def _initialize(self) -> None:
        """初始化Agent Core"""
        try:
            # 初始化大模型
            if HAS_CONFIG:
                self._llm = config.llm_client
                default_logger.info("Agent Core初始化大模型成功")
            else:
                default_logger.warning("config模块未加载，使用模拟大模型")

            # 初始化意图识别器
            try:
                from intent.recognizer import intent_recognizer
                self._intent_recognizer = intent_recognizer
                default_logger.info("意图识别器初始化成功")
            except Exception as e:
                default_logger.error(f"意图识别器初始化失败: {e}")

            # 初始化技能调度器
            self._skill_scheduler = skill_scheduler
            default_logger.info("技能调度器初始化成功")

            # 初始化记忆系统
            try:
                from .memory import memory_system
                self._memory_system = memory_system
                default_logger.info("记忆系统初始化成功")
            except Exception as e:
                default_logger.error(f"记忆系统初始化失败: {e}")

            # 构建状态图
            self._build_graph()

            self._initialized = True
            default_logger.info("Agent Core初始化成功")

        except Exception as e:
            default_logger.error(f"Agent Core初始化失败: {e}")

    def _build_graph(self) -> None:
        """构建LangGraph状态图"""
        # 定义状态图
        self._graph = StateGraph(AgentState)

        # 添加节点
        self._graph.add_node("intent_recognize", self._intent_recognize_node)
        self._graph.add_node("task_decompose", self._task_decompose_node)
        self._graph.add_node("tool_execute", self._tool_execute_node)
        self._graph.add_node("result_aggregate", self._result_aggregate_node)
        self._graph.add_node("error_handle", self._error_handle_node)

        # 添加边（定义流程）
        self._graph.set_entry_point("intent_recognize")

        # 意图识别 -> 任务拆解
        self._graph.add_edge("intent_recognize", "task_decompose")

        # 任务拆解 -> 工具执行
        self._graph.add_conditional_edges(
            "task_decompose",
            self._has_tasks,
            {
                "has_tasks": "tool_execute",
                "no_tasks": "result_aggregate"
            }
        )

        # 工具执行 -> 结果聚合 或 继续执行
        self._graph.add_conditional_edges(
            "tool_execute",
            self._check_execution_status,
            {
                "has_more_tasks": "tool_execute",
                "all_tasks_done": "result_aggregate",
                "has_error": "error_handle"
            }
        )

        # 错误处理 -> 结果聚合
        self._graph.add_edge("error_handle", "result_aggregate")

        # 结果聚合 -> END
        self._graph.add_edge("result_aggregate", END)

        default_logger.info("Agent Core状态图构建完成")

    def _intent_recognize_node(self, state: AgentState) -> Dict[str, Any]:
        """
        意图识别节点

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        try:
            default_logger.info(f"开始意图识别: {state['user_query']}")

            # 执行意图识别
            if self._intent_recognizer:
                intent_result = self._intent_recognizer.predict(state["user_query"])
            else:
                # 模拟意图识别
                intent_result = self._mock_intent_recognize(state["user_query"])

            state["intent"] = intent_result
            state["step_count"] = state.get("step_count", 0) + 1

            # 检查是否需要澄清
            if intent_result.clarification:
                default_logger.info(f"需要澄清意图: {intent_result.clarification}")

            return state

        except Exception as e:
            default_logger.error(f"意图识别失败: {e}")
            state["error"] = f"意图识别失败: {str(e)}"
            return state

    def _task_decompose_node(self, state: AgentState) -> Dict[str, Any]:
        """
        任务拆解节点

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        try:
            intent = state.get("intent")
            if intent is None:
                state["task_list"] = []
                return state

            default_logger.info(f"开始任务拆解，意图: {intent.intent}")

            # 根据意图类型拆解任务
            task_list = self._decompose_tasks(state["user_query"], intent)

            state["task_list"] = task_list
            state["current_task_idx"] = 0
            state["step_count"] = state.get("step_count", 0) + 1

            default_logger.info(f"任务拆解完成，共 {len(task_list)} 个任务")

            return state

        except Exception as e:
            default_logger.error(f"任务拆解失败: {e}")
            state["error"] = f"任务拆解失败: {str(e)}"
            return state

    def _decompose_tasks(
        self,
        query: str,
        intent: IntentResult
    ) -> List[Dict[str, Any]]:
        """
        根据意图拆解任务

        Args:
            query: 用户查询
            intent: 意图结果

        Returns:
            任务列表
        """
        tasks: List[Dict[str, Any]] = []

        if intent.intent == "knowledge_query":
            # 知识查询任务
            tasks = [{
                "task_id": f"task_{len(tasks)}",
                "task_type": "rag_search",
                "description": "检索知识库",
                "params": {
                    "query": query,
                    "business_domain": "finance",
                    "vector_top_k": 20,
                    "keyword_top_k": 20,
                    "final_top_k": 3
                }
            }]

        elif intent.intent == "data_query":
            # 数据查询任务
            tasks = [{
                "task_id": f"task_{len(tasks)}",
                "task_type": "sql_execute",
                "description": "生成并执行SQL查询",
                "params": {
                    "query": query,
                    "table_name": "product_info",
                    "business_description": "查询产品相关数据"
                }
            }]

        elif intent.intent == "task_operation":
            # 任务操作
            tasks = [{
                "task_id": f"task_{len(tasks)}",
                "task_type": "echo",
                "description": "Echo测试",
                "params": {
                    "message": f"执行操作: {query}"
                }
            }]

        else:
            # 未知意图，返回空任务
            default_logger.warning(f"未知意图类型: {intent.intent}")

        return tasks

    def _tool_execute_node(self, state: AgentState) -> Dict[str, Any]:
        """
        工具执行节点

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        try:
            task_list = state.get("task_list", [])
            current_idx = state.get("current_task_idx", 0)

            if current_idx >= len(task_list):
                state["error"] = "任务索引超出范围"
                return state

            current_task = task_list[current_idx]
            task_type = current_task.get("task_type", "")

            default_logger.info(
                f"执行任务 {current_idx + 1}/{len(task_list)}: "
                f"{current_task.get('description', '')}"
            )

            # 执行工具
            if task_type == "rag_search":
                result = self._execute_rag_search(current_task)
            elif task_type == "sql_execute":
                result = self._execute_sql_task(current_task)
            elif task_type == "echo":
                result = self._execute_echo_task(current_task)
            else:
                result = {
                    "success": False,
                    "error": f"未知任务类型: {task_type}"
                }

            # 保存结果
            tool_results = state.get("tool_results", [])
            tool_results.append({
                "task_id": current_task.get("task_id"),
                "task_type": task_type,
                "result": result
            })
            state["tool_results"] = tool_results

            # 更新任务索引
            state["current_task_idx"] = current_idx + 1
            state["step_count"] = state.get("step_count", 0) + 1

            return state

        except Exception as e:
            default_logger.error(f"工具工具执行失败: {e}")
            state["error"] = f"工具工具执行失败: {str(e)}"
            return state

    def _execute_rag_search(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行RAG搜索任务

        Args:
            task: 任务信息

        Returns:
            执行结果
        """
        try:
            request = MCPRequest(
                skill_name="rag_search",
                params=task.get("params", {}),
                trace_id=task.get("task_id")
            )

            if self._skill_scheduler:
                response = self._skill_scheduler.execute(request)
                return response.to_dict()
            else:
                return {
                    "success": False,
                    "error": "技能调度器未初始化"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _execute_sql_task(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行SQL任务

        Args:
            task: 任务信息

        Returns:
            执行结果
        """
        try:
            request = MCPRequest(
                skill_name="sql_execute",
                params=task.get("params", {}),
                trace_id=task.get("task_id")
            )

            if self._skill_scheduler:
                response = self._skill_scheduler.execute(request)
                return response.to_dict()
            else:
                return {
                    "success": False,
                    "error": "技能调度器未初始化"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _execute_echo_task(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行Echo任务

        Args:
            task: 任务信息

        Returns:
            执行结果
        """
        try:
            request = MCPRequest(
                skill_name="echo",
                params=task.get("params", {}),
                trace_id=task.get("task_id")
            )

            if self._skill_scheduler:
                response = self._skill_scheduler.execute(request)
                return response.to_dict()
            else:
                return {
                    "success": False,
                    "error": "技能调度器未初始化"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _result_aggregate_node(self, state: AgentState) -> Dict[str, Any]:
        """
        结果聚合节点

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        try:
            default_logger.info("开始聚合工具结果")

            tool_results = state.get("tool_results", [])
            user_query = state.get("user_query", "")

            # 生成最终答案
            final_answer = self._generate_final_answer(
                user_query,
                tool_results,
                state.get("intent")
            )

            state["final_answer"] = final_answer
            state["step_count"] = state.get("step_count", 0) + 1

            default_logger.info("结果聚合完成")

            return state

        except Exception as e:
            default_logger.error(f"结果聚合失败: {e}")
            state["error"] = f"结果聚合失败: {str(e)}"
            return state

    def _generate_final_answer(
        self,
        query: str,
        tool_results: List[Dict[str, Any]],
        intent: Optional[IntentResult]
    ) -> str:
        """
        生成最终答案

        Args:
            query: 原始查询
            tool_results:结果列表
            intent: 意图结果

        Returns:
            最终答案
        """
        if not tool_results:
            return "抱歉，我无法处理您的请求。"

        # 收集成功和失败的结果
        success_results = []
        failed_results = []

        for result in tool_results:
            if result.get("result", {}).get("success", False):
                success_results.append(result)
            else:
                failed_results.append(result)

        # 简单的答案生成（可以后续优化为LLM生成）
        if not success_results:
            return "抱歉，执行您的请求时遇到了问题。请稍后重试或提供更详细的信息。"

        answer_parts = []
        for result in success_results:
            task_type = result.get("task_type", "")
            tool_result = result.get("result", {})

            if task_type == "rag_search":
                data = tool_result.get("data", {})
                results = data.get("results", [])
                if results:
                    answer_parts.append(f"根据知识库查询: {results[0].get('content', '')[:100]}")

            elif task_type == "sql_execute":
                data = tool_result.get("data", {})
                sql = data.get("sql", "")
                answer_parts.append(f"生成的SQL查询: {sql[:200]}")

            elif task_type == "echo":
                data = tool_result.get("data", {})
                message = data.get("message", "")
                answer_parts.append(message)

        if failed_results:
            error_msgs = [
                r.get("result", {}).get("error", "未知错误")
                for r in failed_results
            ]
            answer_parts.append(f"部分任务失败: {', '.join(error_msgs)}")

        return "\n".join(answer_parts)

    def _error_handle_node(self, state: AgentState) -> Dict[str, Any]:
        """
        错误处理节点

        Args:
            state: 当前状态

        Returns:
            更新后的状态
        """
        try:
            error = state.get("error", "未知错误")
            default_logger.error(f"处理错误: {error}")

            # 检查重试次数
            retry_count = state.get("retry_count", 0)

            if retry_count < MAX_RETRY_COUNT:
                # 可以重试
                state["retry_count"] = retry_count + 1
                state["error"] = None
                default_logger.info(f"准备重试，第 {retry_count + 1} 次")
                return state

            # 超过最大重试次数，生成友好提示
            state["final_answer"] = self._generate_friendly_error(error)
            state["error"] = None

            return state

        except Exception as e:
            default_logger.error(f"错误处理失败: {e}")
            state["final_answer"] = "系统出现错误，请联系管理员。"
            state["error"] = None
            return state

    def _generate_friendly_error(self, error: str) -> str:
        """
        生成友好错误提示

        Args:
            error: 错误信息

        Returns:
            友好提示
        """
        error_lower = error.lower()

        if "not found" in error_lower or "不存在" in error_lower:
            return "抱歉，找不到相关的信息。请检查您的查询是否正确。"
        elif "timeout" in error_lower:
            return "抱歉，请求超时了。请稍后重试。"
        elif "permission" in error_lower or "授权" in error_lower:
            return "抱歉，您没有权限执行此操作。"
        elif "validation" in error_lower or "校验" in error_lower:
            return "抱歉，请求格式不正确。请检查您的输入。"
        else:
            return f"抱歉，处理您的请求时遇到了问题：{error}。请稍后重试。"

    def _has_tasks(self, state: AgentState) -> str:
        """
        检查是否有任务

        Args:
            state: 当前状态

        Returns:
            下一步节点名称
        """
        task_list = state.get("task_list", [])
        return "has_tasks" if task_list else "no_tasks"

    def _check_execution_status(self, state: AgentState) -> str:
        """
        检查执行状态

        Args:
            state: 当前状态

        Returns:
            下一步节点名称
        """
        # 检查是否有错误
        if state.get("error"):
            return "has_error"

        # 检查执行步数
        step_count = state.get("step_count", 0)
        if step_count >= MAX_EXECUTE_STEPS:
            default_logger.warning(f"达到最大执行步数: {MAX_EXECUTE_STEPS}")
            return "all_tasks_done"

        # 检查是否还有任务
        task_list = state.get("task_list", [])
        current_idx = state.get("current_task_idx", 0)

        if current_idx >= len(task_list):
            return "all_tasks_done"
        else:
            return "has_more_tasks"

    def _mock_intent_recognize(self, query: str) -> IntentResult:
        """
        模拟意图识别（用于测试）

        Args:
            query: 用户查询

        Returns:
            意图结果
        """
        from intent.recognizer import IntentResult

        # 简单的关键词匹配
        if any(kw in query for kw in ["什么", "如何", "意思", "介绍", "说明"]):
            return IntentResult(
                intent="knowledge_query",
                confidence=0.8,
                clarification=None
            )
        elif any(kw in query for kw in ["查询", "统计", "多少", "总额", "数量"]):
            return IntentResult(
                intent="data_query",
                confidence=0.8,
                clarification=None
            )
        elif any(kw in query for kw in ["执行", "操作", "创建", "删除", "修改"]):
            return IntentResult(
                intent="task_opperration",
                confidence=0.8,
                clarification=None
            )
        else:
            return IntentResult(
                intent="knowledge_query",
                confidence=0.6,
                clarification="您是要咨询理财产品知识吗？"
            )

    def run(self, query: str) -> Dict[str, Any]:
        """
        运行Agent

        Args:
            query: 用户查询

        Returns:
            执行结果
        """
        if not self._initialized:
            raise RuntimeError("Agent Core未初始化")

        try:
            # 初始化状态
            initial_state = {
                "user_query": query,
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

            # 编译状态图
            compiled_graph = self._graph.compile()

            # 执行状态图
            final_state = compiled_graph.invoke(initial_state)

            default_logger.info("Agent执行完成")

            return {
                "success": not bool(final_state.get("error")),
                "final_answer": final_state.get("final_answer"),
                "steps": final_state.get("step_count", 0),
                "tasks": final_state.get("task_list", []),
                "tool_results": final_state.get("tool_results", []),
                "error": final_state.get("error")
            }

        except Exception as e:
            default_logger.error(f"Agent执行失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "final_answer": "系统出现错误，请稍后重试。"
            }

    def stream_run(self, query: str):
        """
        流式运行Agent

        Args:
            query: 用户查询

        Yields:
            中间结果
        """
        if not self._initialized:
            raise RuntimeError("Agent Core未初始化")

        try:
            # 初始化状态
            initial_state = {
                "user_query": query,
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

            # 编译状态图
            compiled_graph = self._graph.compile()

            # 流式执行
            for event in compiled_graph.stream(initial_state):
                yield event

        except Exception as e:
            default_logger.error(f"Agent流式执行失败: {e}")
            yield {
                "event": "error",
                "error": str(e)
            }


# 创建全局Agent Core实例
agent_core = AgentCore()
