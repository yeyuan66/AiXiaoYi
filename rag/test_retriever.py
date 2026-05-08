"""
RAG检索模块模拟测试（不依赖ES服务）
"""

import json
from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str                # 知识片段内容
    metadata: Dict              # 元数据
    score: float                # 融合得分
    vector_score: Optional[float] = None   # 向量相似度得分
    keyword_score: Optional[float] = None  # 关键词匹配得分
    doc_id: Optional[str] = None          # 文档ID
    chunk_id: Optional[str] = None         # 片段ID


@dataclass
class RetrievalRequest:
    """检索请求参数"""
    query: str                  # 查询文本
    business_domain: Optional[str] = None   # 业务域过滤
    doc_type: Optional[str] = None         # 文档类型过滤
    valid_date: Optional[datetime] = None  # 有效期过滤（查询日期）
    vector_top_k: int = 20       # 向量检索TopK
    keyword_top_k: int = 20      #    关键词检索TopK
    final_top_k: int = 3         # 最终返回TopK
    vector_weight: float = 0.6    # 向量权重
    keyword_weight: float = 0.4    # 关键词权重


# 模拟ES索引结构
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "fields": {
                    "keyword": {
                        "type": "keyword",
                        "ignore_above": 256
                    }
                }
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 512,
                "index": True,
                "similarity": "cosine"
            },
            "metadata": {
                "properties": {
                    "business_domain": {"type": "keyword"},
                    "doc_type": {"type": "keyword"},
                    "valid_from": {"type": "date"},
                    "valid_until": {"type": "date"},
                    "source": {"type": "keyword"},
                    "author": {"type": "keyword"},
                    "version": {"type": "keyword"}
                }
            },
            "created_at": {"type": "date"},
            "updated_at": {"type": "date"}
        }
    }
}


class MockRAGRetriever:
    """模拟RAG检索器（用于测试代码逻辑）"""

    def __init__(self):
        """初始化模拟检索器"""
        self.documents: List[Dict[str, any]] = []

    def init_embedding(self):
        """初始化向量（模拟）"""
        pass

    def insert_documents(self, documents: List[Dict[str, any]]) -> int:
        """插入文档"""
        self.documents.extend(documents)
        return len(documents)

    def retrieve(self, request: RetrievalRequest) -> List[RetrievalResult]:
        """
        混合检索（模拟实现）

        Args:
            request: 检索请求参数

        Returns:
            检索结果列表
        """
        if not request.query or not request.query.strip():
            return []

        # 模拟向量检索（基于简单的关键词匹配）
        vector_results = self._mock_vector_search(request)

        # 模拟关键词检索（BM25）
        keyword_results = self._mock_keyword_search(request)

        # 融合结果
        fused_results = self._fuse_results(
            vector_results,
            keyword_results,
            request.vector_weight,
            request.keyword_weight
        )

        # 按得分排序并返回TopK
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:request.final_top_k]

    def _mock_vector_search(
        self,
        request: RetrievalRequest
    ) -> Dict[str, RetrievalResult]:
        """模拟向量检索"""
        results: Dict[str, RetrievalResult] = {}

        for doc in self.documents:
            # 元数据过滤
            if request.business_domain and \
               doc.get('metadata', {}).get('business_domain') != request.business_domain:
                continue

            if request.doc_type and \
               doc.get('metadata', {}).get('doc_type') != request.doc_type:
                continue

            # 简单的关键词匹配得分
            content = doc.get('content', '')
            query_words = set(request.query.lower().split())
            content_words = set(content.lower().split())

            match_count = len(query_words & content_words)
            if match_count > 0:
                score = match_count / len(query_words) * 0.8 + 0.1

                chunk_id = doc.get('chunk_id', '')
                results[chunk_id] = RetrievalResult(
                    content=content,
                    metadata=doc.get('metadata', {}),
                    score=score,
                    vector_score=score,
                    keyword_score=None,
                    doc_id=doc.get('doc_id'),
                    chunk_id=chunk_id
                )

        return results

    def _mock_keyword_search(
        self,
        request: RetrievalRequest
    ) -> Dict[str, RetrievalResult]:
        """模拟关键词检索"""
        results: Dict[str, RetrievalResult] = {}

        for doc in self.documents:
            # 元数据过滤
            if request.business_domain and \
               doc.get('metadata', {}).get('business_domain') != request.business_domain:
                continue

            if request.doc_type and \
               doc.get('metadata', {}).get('doc_type') != request.doc_type:
                continue

            # BM25-style得分
            content = doc.get('content', '')
            query = request.query.lower()

            if query in content.lower():
                score = 0.9

                chunk_id = doc.get('chunk_id', '')
                results[chunk_id] = RetrievalResult(
                    content=content,
                    metadata=doc.get('metadata', {}),
                    score=score,
                    vector_score=None,
                    keyword_score=score,
                    doc_id=doc.get('doc_id'),
                    chunk_id=chunk_id
                )

        return results

    def _fuse_results(
        self,
        vector_results: Dict[str, RetrievalResult],
        keyword_results: Dict[str, RetrievalResult],
        vector_weight: float,
        keyword_weight: float
    ) -> List[RetrievalResult]:
        """融合结果"""
        fused: Dict[str, RetrievalResult] = {}

        # 归一化得分
        vector_scores = [r.score for r in vector_results.values()]
        keyword_scores = [r.score for r in keyword_results.values()]

        vector_max = max(vector_scores) if vector_scores else 1.0
        keyword_max = max(keyword_scores) if keyword_scores else 1.0

        # 处理向量检索结果
        for chunk_id, result in vector_results.items():
            norm_vector_score = result.score / vector_max

            if chunk_id in keyword_results:
                # 同时出现在两个检索中
                norm_keyword_score = keyword_results[chunk_id].score / keyword_max
                fused_score = (
                    norm_vector_score * vector_weight +
                    norm_keyword_score * keyword_weight
                )
                fused[chunk_id] = RetrievalResult(
                    content=result.content,
                    metadata=result.metadata,
                    score=fused_score,
                    vector_score=result.score,
                    keyword_score=keyword_results[chunk_id].score,
                    doc_id=result.doc_id,
                    chunk_id=chunk_id
                )
            else:
                fused[chunk_id] = RetrievalResult(
                    content=result.content,
                    metadata=result.metadata,
                    score=norm_vector_score * vector_weight,
                    vector_score=result.score,
                    keyword_score=None,
                    doc_id=result.doc_id,
                    chunk_id=chunk_id
                )

        # 处理只出现在关键词检索中的结果
        for chunk_id, result in keyword_results.items():
            if chunk_id not in fused:
                norm_keyword_score = result.score / keyword_max
                fused[chunk_id] = RetrievalResult(
                    content=result.content,
                    metadata=result.metadata,
                    score=norm_keyword_score * keyword_weight,
                    vector_score=None,
                    keyword_score=result.score,
                    doc_id=result.doc_id,
                    chunk_id=chunk_id
                )

        return list(fused.values())


def test_rag_retriever() -> None:
    """测试RAG检索器"""
    print("=" * 50)
    print("AI小益RAG检索模块模拟测试")
    print("=" * 50)

    retriever = MockRAGRetriever()

    # 准备测试文档
    test_documents = [
        {
            "doc_id": "doc_001",
            "chunk_id": "chunk_001",
            "content": "固收类理财产品是指投资于固定收益类资产的理财产品，主要包括债券、存款、票据等。这类产品的风险相对较低，收益相对稳定。",
            "metadata": {
                "business_domain": "finance",
                "doc_type": "product_introduction",
                "valid_from": "2024-01-01",
                "valid_until": "2025-12-31",
                "source": "产品手册"
            }
        },
        {
            "doc_id": "doc_002",
            "chunk_id": "chunk_002",
            "content": "权益类理财产品是指投资于股票、股票型基金等权益类资产的理财产品。这类产品的风险较高，但收益潜力也相对较大。",
            "metadata": {
                "business_domain": "finance",
                "doc_type": "product_introduction",
                "valid_from": "2024-01-01",
                "valid_until": "2025-12-31",
                "source": "产品手册"
            }
        },
        {
            "doc_id": "doc_003",
            "chunk_id": "chunk_003",
            "content": "混合型理财产品是指同时投资于固定收益类资产和权益类资产的理财产品。通过资产配置实现风险和收益的平衡。",
            "metadata": {
                "business_domain": "finance",
                "doc_type": "product_introduction",
                "valid_from": "2024-01-01",
                "valid_until": "2025-12-31",
                "source": "产品手册"
            }
        },
        {
            "doc_id": "doc_004",
            "chunk_id": "chunk_004",
            "content": "理财产品的风险等级通常分为R1（低风险）、R2（中低风险）、R3（中等风险）、R4（中高风险）、R5（高风险）五个等级。",
            "metadata": {
                "business_domain": "finance",
                "doc_type": "risk_level",
                "valid_from": "2024-01-01",
                "valid_until": "2025-12-31",
                "source": "合规文档"
            }
        },
        {
            "doc_id": "doc_005",
            "chunk_id": "chunk_005",
            "content": "净值型理财产品的单位净值是根据产品投资组合的公允价值计算得出的，每日公布。投资者根据单位净值和持有份额计算投资收益。",
            "metadata": {
                "business_domain": "finance",
                "doc_type": "product_features",
                "valid_from": "2024-01-01",
                "valid_until": "2025-12-31",
                "source": "产品说明"
            }
        },
    ]

    print("\n1. 插入测试文档...")
    insert_count = retriever.insert_documents(test_documents)
    print(f"   成功插入 {insert_count} 条文档")

    # 测试混合检索
    print("\n2. 测试混合检索...")
    request = RetrievalRequest(
        query="什么是固收类产品",
        business_domain="finance",
        vector_top_k=20,
        keyword_top_k=20,
        final_top_k=3,
        vector_weight=0.6,
        keyword_weight=0.4
    )

    results = retriever.retrieve(request)

    print(f"\n   查询: {request.query}")
    print(f"   返回 {len(results)} 条结果")
    print("-" * 50)

    for i, result in enumerate(results, 1):
        print(f"\n   结果 {i}:")
        print(f"     内容: {result.content[:80]}...")
        print(f"     融合得分: {result.score:.4f}")
        if result.vector_score is not None:
            print(f"     向量得分: {result.vector_score:.4f}")
        if result.keyword_score is not None:
            print(f"     关键词得分: {result.keyword_score:.4f}")
        print(f"     文档ID: {result.doc_id}")
        print(f"     片段ID: {result.chunk_id}")
        print(f"     元数据: {json.dumps(result.metadata, ensure_ascii=False)}")

    # 测试元数据过滤
    print("\n\n3. 测试元数据过滤...")
    request_filtered = RetrievalRequest(
        query="产品风险等级",
        business_domain="finance",
        doc_type="risk_level",
        vector_top_k=20,
        keyword_top_k=20,
        final_top_k=3
    )

    results_filtered = retriever.retrieve(request_filtered)

    print(f"\n   查询: {request_filtered.query}")
    print(f"   业务域: {request_filtered.business_domain}")
    print(f"   文档类型: {request_filtered.doc_type}")
    print(f"   返回 {len(results_filtered)} 条结果")
    print("-" * 50)

    for i, result in enumerate(results_filtered, 1):
        print(f"\n   结果 {i}:")
        print(f"     内容: {result.content[:80]}...")
        print(f"     得分: {result.score:.4f}")
        print(f"     类型: {result.metadata.get('doc_type')}")

    # 测试不同权重配置
    print("\n\n4. 测试不同权重配置...")
    weight_tests = [
        (0.8, 0.2, "侧重向量"),
        (0.2, 0.8, "侧重关键词"),
        (0.5, 0.5, "均衡权重"),
    ]

    for vw, kw, desc in weight_tests:
        request_weighted = RetrievalRequest(
            query="产品风险",
            business_domain="finance",
            vector_top_k=20,
            keyword_top_k=20,
            final_top_k=2,
            vector_weight=vw,
            keyword_weight=kw
        )

        results_weighted = retriever.retrieve(request_weighted)
        print(f"\n   {desc} (向量={vw}, 关键词={kw}):")
        for i, result in enumerate(results_weighted, 1):
            print(f"     {i}. 得分={result.score:.4f}, "
                  f"内容={result.content[:40]}...")

    # 验证ES索引结构
    print("\n\n5. ES索引结构验证...")
    print("   索引映射结构:")
    print(json.dumps(INDEX_MAPPING, ensure_ascii=False, indent=4))

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


if __name__ == '__main__':
    test_rag_retriever()
