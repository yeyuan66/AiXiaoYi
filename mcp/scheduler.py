"""
工具调度器模块
根据skill_name路由到对应的Skill
"""

from concurrent.futures import ThreadPoolExecutor
from threading import Lock, RLock
from typing import Optional, Dict, Any, Callable, List

from .protocol import MCPRequest, MCPResponse
from .skill_base import SkillBase

try:
    from utils import default_logger
    HAS_LOGGER = True
except ImportError:
    HAS_LOGGER = False
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    default_logger = logging.getLogger('mcp_scheduler')


class SkillScheduler:
    """
    Skill调度器
    负责Skill的注册、发现和调度执行
    """

    _instance: Optional['SkillScheduler'] = None

    def __new__(cls) -> 'SkillScheduler':
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化调度器"""
        if not hasattr(self, '_initialized'):
            self._skills: Dict[str, SkillBase] = {}
            self._lock = Lock()  # 写锁
            self._read_lock = RLock()  # 读锁
            self._executor: Optional[ThreadPoolExecutor] = None
            self._max_workers = 10

            self._initialized = True
            default_logger.info("Skill调度器初始化成功")

            # 注册默认技能
            self._register_default_skills()

    def _register_default_skills(self) -> None:
        """注册默认技能"""
        try:
            from .skills.sql_skill import SQLSkill
            from .skills.rag_skill import RAGSearchSkill

            # 注册SQL执行技能
            sql_skill = SQLSkill()
            self.register_skill(sql_skill)

            # 注册RAG检索技能
            rag_skill = RAGSearchSkill()
            self.register_skill(rag_skill)

            default_logger.info("默认技能注册完成")
        except Exception as e:
            default_logger.error(f"默认技能注册失败: {e}")

    def register_skill(self, skill: SkillBase) -> bool:
        """
        注册Skill

        Args:
            skill: Skill实例

        Returns:
            是否注册成功
        """
        with self._lock:
            name = skill.name

            if name in self._skills:
                default_logger.warning(
                    f"Skill {name} 已存在，将被覆盖"
                )

            self._skills[name] = skill
            default_logger.info(f"Skill注册成功: {skill}")
            return True

    def unregister_skill(self, name: str) -> bool:
        """
        注销Skill

        Args:
            name: Skill名称

        Returns:
            是否注销成功
        """
        with self._lock:
            if name in self._skills:
                del self._skills[name]
                default_logger.info(f"Skill注销成功: {name}")
                return True
            else:
                default_logger.warning(f"Skill不存在: {name}")
                return False

    def get_skill(self, name: str) -> Optional[SkillBase]:
        """
        获取Skill

        Args:
            name: Skill名称

        Returns:
            Skill实例或None
        """
        with self._read_lock:
            return self._skills.get(name)

    def list_skills(self) -> Dict[str, Dict[str, Any]]:
        """
        列出所有已注册的Skill

        Returns:
            Skill信息字典
        """
        with self._read_lock:
            return {
                name: skill.get_info()
                for name, skill in self._skills.items()
            }

    def has_skill(self, name: str) -> bool:
        """
        检查Skill是否已注册

        Args:
            name: Skill名称

        Returns:
            是否存在
        """
        with self._read_lock:
            return name in self._skills

    def execute(
        self,
        request: MCPRequest
    ) -> MCPResponse:
        """
        执行Skill（同步）

        Args:
            request: MCP请求

        Returns:
            MCP响应
        """
        # 验证请求格式
        is_valid, error_msg = request.validate()
        if not is_valid:
            return MCPResponse.error_response(
                error=f"请求格式错误: {error_msg}",
                trace_id=request.trace_id,
                request_id=request.request_id
            )

        skill_name = request.skill_name

        # 检查Skill是否存在
        with self._read_lock:
            skill = self._skills.get(skill_name)

        if skill is None:
            default_logger.error(
                f"Skill不存在: {skill_name}, trace_id: {request.trace_id}"
            )
            return MCPResponse.error_response(
                error=f"Skill不存在: {skill_name}",
                trace_id=request.trace_id,
                request_id=request.request_id
            )

        # 健康检查
        if not skill.health_check():
            default_logger.error(
                f"Skill不健康: {skill_name}, trace_id: {request.trace_id}"
            )
            return MCPResponse.error_response(
                error=f"Skill不健康: {skill_name}",
                trace_id=request.trace_id,
                request_id=request.request_id
            )

        default_logger.info(
            f"开始执行Skill: {skill_name}, trace_id: {request.trace_id}"
        )

        # 执行Skill
        response = skill._execute_safe(request)

        if response.success:
            default_logger.info(
                f"Skill执行成功: {skill_name}, "
                f"trace_id: {request.trace_id}, "
                f"耗时: {response.execution_time:.2f}ms"
            )
        else:
            default_logger.error(
                f"Skill执行失败: {skill_name}, "
                f"trace_id: {request.trace_id}, "
                f"错误: {response.error}"
            )

        return response

    def execute_async(
        self,
        request: MCPRequest,
        callback: Optional[Callable[[MCPResponse], None]] = None
    ) -> str:
        """
        异步执行Skill

        Args:
            request: MCP请求
            callback: 回调函数（可选）

        Returns:
            执行任务ID
        """
        import uuid
        task_id = str(uuid.uuid4())

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        # 提交任务
        future = self._executor.submit(self.execute, request)

        # 添加回调
        if callback is not None:
            def _callback_wrapper(future: 'Future'):
                try:
                    response = future.result()
                    callback(response)
                except Exception as e:
                    default_logger.error(f"回调执行失败: {e}")
                    # 返回错误响应
                    error_response = MCPResponse.error_response(
                        error=f"回调执行失败: {str(e)}",
                        trace_id=request.trace_id,
                        request_id=request.request_id
                    )
                    callback(error_response)

            future.add_done_callback(_callback_wrapper)

        default_logger.info(
            f"异步执行任务提交成功: {task_id}, "
            f"skill: {request.skill_name}, trace_id: {request.trace_id}"
        )

        return task_id

    def batch_execute(
        self,
        requests: List[MCPRequest]
    ) -> List[MCPResponse]:
        """
        批量执行Skill

        Args:
            requests: MCP请求列表

        Returns:
            MCP响应列表
        """
        responses = []

        for request in requests:
            response = self.execute(request)
            responses.append(response)

        return responses

    def batch_execute_async(
        self,
        requests: List[MCPRequest],
        callback: Optional[Callable[[List[MCPResponse]], None]] = None
    ) -> list[str]:
        """
        批量异步执行Skill

        Args:
            requests: MCP请求列表
            callback: 回调函数（可选）

        Returns:
            执行任务ID列表
        """
        task_ids = []

        # 收集所有响应
        responses = []
        completed_count = [0]

        def _individual_callback(response: MCPResponse) -> None:
            responses.append(response)
            completed_count[0] += 1

            # 所有请求完成后调用总回调
            if completed_count[0] == len(requests) and callback is not None:
                callback(responses)

        # 提交所有任务
        for request in requests:
            task_id = self.execute_async(request, _individual_callback)
            task_ids.append(task_id)

        return task_ids

    def discover_skills(
        self,
        module_name: Optional[str] = None
    ) -> int:
        """
        自动发现并注册Skill

        Args:
            module_name: 模块名称（可选），不传则发现mcp.skills模块下的所有Skill

        Returns:
            注册成功的Skill数量
        """
        import importlib
        import inspect

        discovered_count = 0

        try:
            if module_name is None:
                # 发现mcp.skills模块
                module_name = "mcp.skills"

            module = importlib.import_module(module_name)

            # 遍历模块中的所有成员
            for name, obj in inspect.getmembers(module):
                # 检查是否是SkillBase的子类且不是SkillBase本身
                if (inspect.isclass(obj) and
                    issubclass(obj, SkillBase) and
                    obj != SkillBase):
                    try:
                        # 实例化并注册
                        skill_instance = obj()
                        if self.register_skill(skill_instance):
                            discovered_count += 1
                    except Exception as e:
                        default_logger.error(
                            f"Skill实例化失败: {name}, 错误: {e}"
                        )

            default_logger.info(
                f"Skill自动发现完成，注册成功: {discovered_count}个"
            )
            return discovered_count

        except Exception as e:
            default_logger.error(f"Skill自动发现失败: {e}")
            return discovered_count

    def health_check(self) -> Dict[str, Any]:
        """
        调度器健康检查

        Returns:
            健康状态字典
        """
        with self._read_lock:
            skill_count = len(self._skills)
            healthy_count = sum(
                1 for skill in self._skills.values()
                if skill.health_check()
            )

            return {
                "status": "healthy" if healthy_count == skill_count else "degraded",
                "total_skills": skill_count,
                "healthy_skills": healthy_count,
                "unhealthy_skills": skill_count - healthy_count,
                "details": {
                    name: skill.health_check()
                    for name, skill in self._skills.items()
                }
            }

    def shutdown(self) -> None:
        """关闭调度器，清理资源"""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

        with self._lock:
            self._skills.clear()

        default_logger.info("Skill调度器已关闭")


# 创建全局调度器实例
skill_scheduler = SkillScheduler()
