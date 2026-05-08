"""
SQL Skill实现
整合SQL生成和校验逻辑
"""

from typing import Dict, Any, Optional

from ..skill_base import SkillBase
from ..protocol import MCPRequest, MCPResponse

try:
    from codeact.generator import SQLGenerator, SQLGenerationRequest
    from codeact.validator import SQLValidator, ValidationResult
    HAS_CODEACT = True
except ImportError:
    HAS_CODEACT = False


class SQLSkill(SkillBase):
    """
    SQL Skill
    整合SQL生成和校验逻辑，生成安全的SQL查询
    """

    def __init__(self):
        """初始化SQL Skill"""
        self._generator: Optional['SQLGenerator'] = None
        self._validator: Optional['SQLValidator'] = None

        super().__init__(
            name="sql_execute",
            version="1.0.0"
        )

    def _init_components(self) -> None:
        """初始化组件"""
        if not HAS_CODEACT:
            return

        try:
            from codeact.generator import sql_generator
            from codeact.validator import sql_validator
            self._generator = sql_generator
            self._validator = sql_validator
        except Exception as e:
            from utils import default_logger
            default_logger.error(f"组件初始化失败: {e}")

    def _initialize(self) -> None:
        """初始化Skill"""
        super()._initialize()

        self._init_components()

        if self._generator is None or self._validator is None:
            from utils import default_logger
            default_logger.warning("组件未初始化，Skill无法正常工作")

    def execute(self, request: MCPRequest) -> MCPResponse:
        """
        执行SQL生成和校验

        Args:
            request: MCP请求
                params:
                    query: 用户自然语言查询（必需）
                    table_name: 主表名称（必需）
                    business_description: 业务口径（可选）
                    use_few_shot: 是否使用Few-shot（可选，默认True）
                    strict_validation: 严格校验（可选，默认True）
                    return_sql_only: 只返回SQL（可选，默认False）

        Returns:
            MCP响应
                data:
                    sql: 生成的SQL
                    explanation: SQL解释
                    is_valid: 是否通过校验
                    validation_result: 校验结果详情
                    execution_result: 执行结果（安全沙箱部分省略）
        """
        params = request.params

        # 验证必需参数
        is_valid, error_msg = self.validate_params(
            params,
            required_params=["query", "table_name"]
        )
        if not is_valid:
            return MCPResponse.error_response(
                error=f"参数验证失败: {error_msg}",
                trace_id=request.trace_id,
                request_id=request.request_id
            )

        # 检查组件是否可用
        if self._generator is None or self._validator is None:
            return MCPResponse.error_response(
                error="SQL组件未初始化",
                trace_id=request.trace_id,
                request_id=request.request_id
            )

        try:
            # 1. 生成SQL
            from datetime import datetime

            # 获取Few-shot示例
            few_shot_examples = None
            if params.get("use_few_shot", True):
                few_shot_examples = self._generator.retrieve_few_shot_examples(
                    query=params.get("query", ""),
                    table_name=params.get("table_name", ""),
                    top_k=3
                )

            # 构建生成请求
            gen_request = SQLGenerationRequest(
                user_query=params.get("query", ""),
                table_name=params.get("table_name", ""),
                business_description=params.get("business_description"),
                few_shot_examples=few_shot_examples,
                current_date=datetime.now()
            )

            gen_result = self._generator.generate(gen_request)

            if not gen_result.success:
                return MCPResponse.error_response(
                    error=f"SQL生成失败: {gen_result.error}",
                    trace_id=request.trace_id,
                    request_id=request.request_id
                )

            # 2. 校验SQL
            strict_validation = params.get("strict_validation", True)
            validation_result = self._validator.validate(gen_result.sql)

            # 3. 如果校验失败且要求严格校验，返回错误
            if not validation_result.is_valid and strict_validation:
                return MCPResponse.error_response(
                    error=f"SQL校验失败: {validation_result.error_message}",
                    trace_id=request.trace_id,
                    request_id=request.request_id
                )

            # 4. 执行SQL（安全沙箱部分省略，只返回校验结果）
            execution_result = {
                "status": "skipped",
                "message": "安全沙箱执行已禁用，仅返回SQL和校验结果"
            }

            # 5. 返回结果
            return_only_sql = params.get("return_sql_only", False)

            if return_only_sql:
                response_data = {
                    "sql": gen_result.sql
                }
            else:
                response_data = {
                    "sql": gen_result.sql,
                    "explanation": gen_result.explanation,
                    "is_valid": validation_result.is_valid,
                    "validation_result": {
                        "is_valid": validation_result.is_valid,
                        "error_message": validation_result.error_message,
                        "error_code": validation_result.error_code,
                        "warnings": validation_result.warnings or []
                    },
                    "execution_result": execution_result
                }

            return MCPResponse.success_response(
                data=response_data,
                trace_id=request.trace_id,
                request_id=request.request_id
            )

        except Exception as e:
            return MCPResponse.error_response(
                error=f"SQL执行失败: {str(e)}",
                trace_id=request.trace_id,
                request_id=request.request_id
            )

    def get_description(self) -> str:
        """
        获取Skill描述

        Returns:
            Skill描述
        """
        return """SQL Skill：整合SQL生成和校验逻辑，生成安全的SQL查询。
基于GLM4.7生成SQL，使用sqlglot进行AST语法校验，
拦截增删改操作、跨库JOIN、嵌套查询超过3层等不安全操作。"""

    def get_param_schema(self) -> Dict[str, Any]:
        """
        获取参数schema

        Returns:
            参数schema
        """
        return {
            "query": {
                "type": "string",
                "required": True,
                "description": "用户自然语言查询"
            },
            "table_name": {
                "type": "string",
                "required": True,
                "description": "主表名称（如：product_info）"
            },
            "business_description": {
                "type": "string",
                "required": False,
                "description": "业务口径说明"
            },
            "use_few_shot": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "是否使用Few-shot学习"
            },
            "strict_validation": {
                "type": "boolean",
                "required": False,
                "default": True,
                "description": "是否启用严格校验（校验失败则拒绝执行）"
            },
            "return_sql_only": {
                "type": "boolean",
                "required": False,
                "default": False,
                "description": "是否只返回SQL（不包含校验和执行结果）"
            }
        }

    def validate_sql_only(
        self,
        sql: str
    ) -> Dict[str, Any]:
        """
        只校验SQL（不生成）

        Args:
            sql: SQL语句

        Returns:
            校验结果字典
        """
        if self._validator is None:
            return {
                "is_valid": False,
                "error_message": "SQL校验器未初始化"
            }

        try:
            result = self._validator.validate(sql)
            return {
                "is_valid": result.is_valid,
                "error_message": result.error_message,
                "error_code": result.error_code,
                "warnings": result.warnings or [],
                "details": result.details
            }
        except Exception as e:
            return {
                "is_valid": False,
                "error_message": f"校验失败: {str(e)}"
            }

    def generate_sql_only(
        self,
        query: str,
        table_name: str,
        business_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        只生成SQL（不校验）

        Args:
            query: 用户查询
            table_name: 表名
            business_description: 业务口径

        Returns:
            生成结果字典
        """
        if self._generator is None:
            return {
                "success": False,
                "error": "SQL生成器未初始化"
            }

        try:
            from datetime import datetime
            from codeact.generator import SQLGenerationRequest

            gen_request = SQLGenerationRequest(
                user_query=query,
                table_name=table_name,
                business_description=business_description,
                few_shot_examples=None,
                current_date=datetime.now()
            )

            result = self._generator.generate(gen_request)

            return {
                "success": result.success,
                "sql": result.sql,
                "explanation": result.explanation,
                "parameters": result.parameters,
                "error": result.error
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"生成失败: {str(e)}"
            }
