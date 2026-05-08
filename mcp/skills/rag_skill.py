"""
RAG Skill实现
调用RAGRetriever进行知识检索
"""

from typing import Dict, Any, Optional

from ..skill_base import SkillBase
from ..protocol import MCPRequest, MCPResponse

try:
    from rag.retriever import RAGRetriever, RetrievalRequest
    HAS_RAG = True
except ImportError:
    HAS_RAG = False


class RAGSearchSkill(SkillBase):
    """
    RAG检索Skill
    调用RAGRetriever进行混合检索
    """

    def __init__(self):
        """初始化RAG检索Skill"""
        self._rag_retriever: Optional['RAGRetriever'] = None

        super().__init__(
            name="rag_search",
            version="1.0.0"
        )

    def _init_rag(self) -> None:
        """初始化RAG检索器"""
        if not HAS_RAG:
            return

        try:
            from rag.retriever import rag_retriever
            self._rag_retriever = rag_retriever
        except Exception as e:
            from utils import default_logger
            default_logger.error(f"RAG检索器初始化失败: {e}")

    def _initialize(self) -> None:
        """初始化Skill"""
        super()._initialize()

        self._init_rag()

        if self._rag_retriever is None:
            from utils import default_logger
            default_logger.warning("RAG检索器未初始化，Skill无法正常工作")

    def execute(self, request: MCPRequest) -> MCPResponse:
        """
        执行RAG检索

        Args:
            request: MCP请求
                params:
                    query: 查询文本（必需）
                    business_domain: 业务域（可选）
                    doc_type: 文档类型（可选）
                    vector_top_k: 向量检索TopK（可选，默认20）
                    keyword_top_k: 关键词检索TopK（可选，默认20）
                    final_top_k: 最终返回TopK（可选，默认3）
                    vector_weight: 向量权重（可选，默认0.6）
                    keyword_weight: 关键词权重（可选，默认0.4）

        Returns:
            MCP响应
                data:
                    results: 检索结果列表
                        - content: 内容
                        - metadata: 元数据
                        - score: 融合得分
                        - vector_score: 向量得分
                        - keyword_score: 关键词得分
                        - doc_id: 文档ID
                        - chunk_id: 片段ID
                    query: 原始查询
                    total: 结果数量
        """
        params = request.params

        # 验证必需参数
        is_valid, error_msg = self.validate_params(
            params,
            required_params=["query"]
        )
        if not is_valid:
            return MCPResponse.error_response(
                error=f"参数验证失败: {error_msg}",
                trace_id=request.trace_id,
                request_id=request.request_id
            )

        # 检查RAG检索器是否可用
        if self._rag_retriever is None:
            return MCPResponse.error_response(
                error="RAG检索器未初始化",
                trace_id=request.trace_id,
                request_id=request.request_id
            )

        try:
            # 构建检索请求
            from datetime import datetime
            retrieval_request = RetrievalRequest(
                query=params.get("query", ""),
                business_domain=params.get("business_domain"),
                doc_type=params.get("doc_type"),
                # 简单处理有效期
                valid_date=None,
                vector_top_k=params.get("vector_top_k", 20),
                keyword_top_k=params.get("keyword_top_k", 20),
                final_top_k=params.get("final_top_k", 3),
                vector_weight=params.get("vector_weight", 0.6),
                keyword_weight=params.get("keyword_weight", 0.4)
            )

            # 执行检索
            results = self._rag_retriever.retrieve(retrieval_request)

            # 转换结果格式
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.content,
                    "metadata": result.metadata,
                    "score": result.score,
                    "vector_score": result.vector_score,
                    "keyword_score": result.keyword_score,
                    "doc_id": result.doc_id,
                    "chunk_id": result.chunk_id
                })

            # 返回成功响应
            return MCPResponse.success_response(
                data={
                    "results": formatted_results,
                    "query": retrieval_request.query,
                    "total": len(formatted_results)
                },
                trace_id=request.trace_id,
                request_id=request.request_id
            )

        except Exception as e:
            return MCPResponse.error_response(
                error=f"检索执行失败: {str(e)}",
                trace_id=request.trace_id,
                request_id=request.request_id
            )

    def get_description(self) -> str:
        """
        获取Skill描述

        Returns:
            Skill描述
        """
        return """RAG检索Skill：基于Elasticsearch实现混合检索（向量相似度 + BM25关键词）
支持元数据过滤和加权融合，返回最相关的知识片段。"""

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
                "description": "查询文本"
            },
            "business_domain": {
                "type": "string",
                "required": False,
                "description": "业务域过滤（如：finance）"
            },
            "doc_type": {
                "type": "string",
                "required": False,
                "description": "文档类型过滤（如：product_introduction, risk_level）"
            },
            "vector_top_k": {
                "type": "integer",
                "required": False,
                "default": 20,
                "description": "向量检索TopK"
            },
            "keyword_top_k": {
                "type": "integer",
                "required": False,
                "default": 20,
                "description": "关键词检索TopK"
            },
            "final_top_k": {
                "type": "integer",
                "required": False,
                "default": 3,
                "description": "最终返回TopK"
            },
            "vector_weight": {
                "type": "float",
                "required": False,
                "default": 0.6,
                "description": "向量权重（0.0-1.0）"
            },
            "keyword_weight": {
                "type": "float",
                "required": False,
                "default": 0.4,
                "description": "关键词权重（0.0-1.0）"
            }
        }
