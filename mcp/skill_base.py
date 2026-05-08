"""
Skill基类模块
定义所有Skill的通用接口和基础功能
"""

import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
from threading import Lock

from .protocol import MCPRequest, MCPResponse, MCPError


class SkillBase(ABC):
    """
    Skill基类
    所有Skill必须继承自该类并实现execute方法
    保证线程安全
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        """
        初始化Skill

        Args:
            name: Skill名称
            version: Skill版本
        """
        self._name = name
        self._version = version
        self._lock = Lock()  # 线程安全锁
        self._initialized = False

        # 初始化Skill
        self._initialize()
        self._initialized = True

    @property
    def name(self) -> str:
        """Skill名称"""
        return self._name

    @property
    def version(self) -> str:
        """Skill版本"""
        return self._version

    @abstractmethod
    def execute(self, request: MCPRequest) -> MCPResponse:
        """
        执行Skill（必须由子类实现）

        Args:
            request: MCP请求

        Returns:
            MCP响应
        """
        pass

    def _initialize(self) -> None:
        """
        初始化Skill（可由子类重写）
        """
        pass

    def _execute_safe(self, request: MCPRequest) -> MCPResponse:
        """
        线程安全的执行方法

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

        start_time = time.time()

        try:
            # 获取锁，确保线程安全
            with self._lock:
                # 执行具体的Skill逻辑
                response = self.execute(request)

            # 计算执行时间
            execution_time = (time.time() - start_time) * 1000
            response.execution_time = execution_time

            # 确保trace_id一致
            if response.trace_id != request.trace_id:
                response.trace_id = request.trace_id

            return response

        except Exception as e:
            # 异常处理
            execution_time = (time.time() - start_time) * 1000

            return MCPResponse.error_response(
                error=f"Skill执行失败: {str(e)}",
                trace_id=request.trace_id,
                request_id=request.request_id,
                execution_time=execution_time
            )

    def validate_params(
        self,
        params: Dict[str, Any],
        required_params: list[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        验证参数

        Args:
            params: 参数字典
            required_params: 必需参数列表

        Returns:
            (是否有效, 错误信息)
        """
        for param in required_params:
            if param not in params:
                return False, f"缺少必需参数: {param}"

        return True, None

    def get_info(self) -> Dict[str, Any]:
        """
        获取Skill信息

        Returns:
            Skill信息字典
        """
        return {
            "name": self._name,
            "version": self._version,
            "initialized": self._initialized,
            "description": self.get_description()
        }

    @abstractmethod
    def get_description(self) -> str:
        """
        获取Skill描述（必须由子类实现）

        Returns:
            Skill描述
        """
        pass

    @abstractmethod
    def get_param_schema(self) -> Dict[str, Any]:
        """
        获取参数schema（必须由子类实现）

        Returns:
            参数schema字典
        """
        pass

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            是否健康
        """
        return self._initialized

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self._name}@{self._version}"
