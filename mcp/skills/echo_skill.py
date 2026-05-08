"""
Echo Skill实现
测试用Skill，返回输入参数
"""

from typing import Dict, Any

from ..skill_base import SkillBase
from ..protocol import MCPRequest, MCPResponse


class EchoSkill(SkillBase):
    """
    Echo测试Skill
    简单返回输入参数，用于测试MCP协议
    """

    def __init__(self):
        """初始化Echo Skill"""
        super().__init__(
            name="echo",
            version="1.0.0"
        )

    def execute(self, request: MCPRequest) -> MCPResponse:
        """
        执行Echo操作

        Args:
            request: MCP请求
                params:
                    message: 要回显的消息（可选）
                    delay: 延迟时间（毫秒，可选，用于测试超时）

        Returns:
            MCP响应
                data:
                    message: 回显消息
                    original_params: 原始参数
                    timestamp: 服务器时间戳
        """
        params = request.params

        try:
            # 处理延迟（用于测试）
            delay = params.get("delay", 0)
            if delay and delay > 0:
                import time
                time.sleep(delay / 1000.0)

            # 获取消息
            message = params.get("message", "Hello from EchoSkill!")

            # 返回成功响应
            import time
            return MCPResponse.success_response(
                data={
                    "message": message,
                    "original_params": params,
                    "timestamp": time.time()
                },
                trace_id=request.trace_id,
                request_id=request.request_id
            )

        except Exception as e:
            return MCPResponse.error_response(
                error=f"Echo执行失败: {str(e)}",
                trace_id=request.trace_id,
                request_id=request.request_id
            )

    def get_description(self) -> str:
        """
        获取Skill描述

        Returns:
            Skill描述
        """
        return """Echo测试Skill：简单返回输入参数，用于测试MCP协议和Skill调度器。"""

    def get_param_schema(self) -> Dict[str, Any]:
        """
        获取参数schema

        Returns:
            参数schema
        """
        return {
            "message": {
                "type": "string",
                "required": False,
                "default": "Hello from EchoSkill!",
                "description": "要回显的消息"
            },
            "delay": {
                "type": "integer",
                "required": False,
                "default": 0,
                "description": "延迟时间（毫秒）"
            }
        }
