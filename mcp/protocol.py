"""
MCP协议定义模块
定义MCP请求和响应的数据结构
"""

import json
import uuid
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union, Tuple
from enum import Enum


class RequestStatus(Enum):
    """请求状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class MCPRequest:
    """
    MCP请求格式

    Args:
        skill_name: 技能名称
        params: 参数字典
        trace_id: 追踪ID（可选，不传则自动生成）
        request_id: 请求ID（可选，不传则自动生成）
        timestamp: 请求时间戳（可选，不传则自动生成）
        context: 上下文信息（可选）
    """
    skill_name: str
    params: Dict[str, Any]
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    timestamp: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """初始化后处理，生成默认值"""
        if self.trace_id is None:
            self.trace_id = str(uuid.uuid4())

        if self.request_id is None:
            self.request_id = str(uuid.uuid4())

        if self.timestamp is None:
            import time
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "skill_name": self.skill_name,
            "params": self.params,
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "context": self.context
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPRequest':
        """从字典创建MCPRequest"""
        return cls(
            skill_name=data.get("skill_name", ""),
            params=data.get("params", {}),
            trace_id=data.get("trace_id"),
            request_id=data.get("request_id"),
            timestamp=data.get("timestamp"),
            context=data.get("context")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'MCPRequest':
        """从JSON字符串创建MCPRequest"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        验证请求格式

        Returns:
            (是否有效, 错误信息)
        """
        if not self.skill_name or not self.skill_name.strip():
            return False, "skill_name不能为空"

        if not isinstance(self.params, dict):
            return False, "params必须是字典类型"

        if self.trace_id is None:
            return False, "trace_id不能为空"

        return True, None


@dataclass
class MCPResponse:
    """
    MCP响应格式

    Args:
        success: 是否成功
        data: 响应数据
        error: 错误信息
        trace_id: 追踪ID
        request_id: 请求ID
        status: 响应状态
        timestamp: 响应时间戳（可选，不传则自动生成）
        execution_time: 执行时间（毫秒，可选）
        context: 上下文信息（可选）
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    status: Optional[str] = None
    timestamp: Optional[float] = None
    execution_time: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """初始化后处理"""
        if self.status is None:
            self.status = RequestStatus.SUCCESS.value if self.success else RequestStatus.FAILED.value

        if self.timestamp is None:
            import time
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "success": self.success,
            "trace_id": self.trace_id,
            "request_id": self.request_id,
            "status": self.status,
            "timestamp": self.timestamp
        }

        if self.data is not None:
            result["data"] = self.data

        if self.error is not None:
            result["error"] = self.error

        if self.execution_time is not None:
            result["execution_time"] = self.execution_time

        if self.context is not None:
            result["context"] = self.context

        return result

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPResponse':
        """从字典创建MCPResponse"""
        return cls(
            success=data.get("success", False),
            data=data.get("data"),
            error=data.get("error"),
            trace_id=data.get("trace_id"),
            request_id=data.get("request_id"),
            status=data.get("status"),
            timestamp=data.get("timestamp"),
            execution_time=data.get("execution_time"),
            context=data.get("context")
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'MCPResponse':
        """从JSON字符串创建MCPResponse"""
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def success_response(
        cls,
        data: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
        execution_time: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> 'MCPResponse':
        """
        创建成功响应

        Args:
            data: 响应数据
            trace_id: 追踪ID
            request_id: 请求ID
            execution_time: 执行时间（毫秒）
            context: 上下文信息

        Returns:
            MCPResponse实例
        """
        return cls(
            success=True,
            data=data,
            trace_id=trace_id,
            request_id=request_id,
            execution_time=execution_time,
            context=context
        )

    @classmethod
    def error_response(
        cls,
        error: str,
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
        execution_time: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> 'MCPResponse':
        """
        创建错误响应

        Args:
            error: 错误信息
            trace_id: 追踪ID
            request_id: 请求ID
            execution_time: 执行时间（毫秒）
            context: 上下文信息

        Returns:
            MCPResponse实例
        """
        return cls(
            success=False,
            error=error,
            trace_id=trace_id,
            request_id=request_id,
            execution_time=execution_time,
            context=context
        )

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        验证响应格式

        Returns:
            (是否有效, 错误信息)
        """
        if self.trace_id is None:
            return False, "trace_id不能为空"

        if not isinstance(self.success, bool):
            return False, "success必须是布尔类型"

        if self.success and self.data is None:
            return False, "成功响应必须包含data"

        if not self.success and self.error is None:
            return False, "失败响应必须包含error"

        return True, None


@dataclass
class MCPError:
    """
    MCP错误定义

    Args:
        code: 错误代码
        message: 错误消息
        detail: 错误详情（可选）
    """
    code: str
    message: str
    detail: Optional[str] = None

    # 预定义错误代码
    INVALID_REQUEST = "INVALID_REQUEST"
    SKILL_NOT_FOUND = "SKILL_NOT_FOUND"
    SKILL_EXECUTION_ERROR = "SKILL_EXECUTION_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    AUTHORIZATION_ERROR = "AUTHORIZATION_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "code": self.code,
            "message": self.message
        }
        if self.detail is not None:
            result["detail"] = self.detail
        return result

    def __str__(self) -> str:
        """字符串表示"""
        return f"[{self.code}] {self.message}"


class MCPProtocolValidator:
    """
    MCP协议验证器
    用于验证请求和响应的格式
    """

    @staticmethod
    def validate_request(request: MCPRequest) -> tuple[bool, Optional[str]]:
        """
        验证MCP请求

        Args:
            request: MCPRequest实例

        Returns:
            (是否有效, 错误信息)
        """
        return request.validate()

    @staticmethod
    def validate_response(response: MCPResponse) -> tuple[bool, Optional[str]]:
        """
        验证MCP响应

        Args:
            response: MCPResponse实例

        Returns:
            (是否有效, 错误信息)
        """
        return response.validate()

    @staticmethod
    def validate_trace_id_consistency(
        request: MCPRequest,
        response: MCPResponse
    ) -> tuple[bool, Optional[str]]:
        """
        验证请求和响应的trace_id一致性

        Args:
            request: MCPRequest实例
            response: MCPResponse实例

        Returns:
            (是否一致, 错误信息)
        """
        if request.trace_id != response.trace_id:
            return False, f"trace_id不一致: 请求={request.trace_id}, 响应={response.trace_id}"
        return True, None


# = 导出 = =
__all__ = [
    'MCPRequest',
    'MCPResponse',
    'MCPError',
    'MCPProtocolValidator',
    'RequestStatus'
]
