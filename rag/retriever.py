"""
RAG知识库检索模块
基于Elasticsearch7.17实现混合检索（向量相似度 + BM25关键词）
"""

import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

from elasticsearch import Elasticsearch

try:
    from config import config
    from utils import default_logger
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('rag_retriever')
    default_logger = logger


# 向量模型配置
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
EMBEDDING_DEVICE = "cpu"  # 或 "cuda" 如果有GPU

# ES索引结构
INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "doc_id": {
                "type": "keyword"
            },
            "chunk_id": {
                "type": "keyword"
            },
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
                    "business_domain": {
                        "type": "keyword"
                    },
                    "doc_type": {
                        "type": "keyword"
                    },
                    "valid_from": {
                        "type": "date"
                    },
                    "valid_until": {
                        "type": "date"
                    },
                    "source": {
                        "type": "keyword"
                    },
                    "author": {
                        "type": "keyword"
                    },
                    "version": {
                        "type": "keyword"
                    }
                }
            },
            "created_at": {
                "type": "date"
            },
            "updated_at": {
                "type": "date"
            }
        }
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "analysis": {
            "analyzer": {
                "ik_max_word": {
                    "type": "custom",
                    "tokenizer": "ik_max_word"
                }
            }
        }
    }
}


@dataclass
class RetrievalResult:
    """检索结果"""
    content: str                # 知识片段内容
    metadata: dict              # 元数据
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
    keyword_top_k: int = 20      # 关键词检索TopK
    final_top_k: int = 3         # 最终返回TopK
    vector_weight: float = 0.6    # 向量权重
    keyword_weight: float = 0.4    # 关键词权重


class RAGRetriever:
    """
    RAG检索器单例类
    基于Elasticsearch实现混合检索（向量相似度 + BM25关键词）
    """

    _instance: Optional['RAGRetriever'] = None

    def __new__(cls) -> 'RAGRetriever':
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化RAG检索器"""
        if not hasattr(self, '_initialized'):
            self.es_client: Optional[Elasticsearch] = None
            self.index_name: str = ""
            self._embedding_model = None
            self._initialized = False

            self._initialize()

    def _initialize(self) -> None:
        """初始化ES客户端和向量模型"""
        try:
            # 初始化ES客户端
            if HAS_CONFIG:
                self.es_client = config.es_client
                self.index_name = config.es_index
            else:
                # 开发环境配置
                self.es_client = Elasticsearch(
                    ["http://localhost:9200"],
                    timeout=30,
                    max_retries=3,
                    verify_certs=False
                )
                self.index_name = "finance_rag_knowledge_v1"

            # 测试ES连接
            if not self.es_client.ping():
                raise ConnectionError("无法连接到Elasticsearch")

            # 初始化向量模型
            self._init_embedding_model()

            default_logger.info(
                f"RAG检索器初始化成功，索引: {self.index_name}"
            )
            self._initialized = True

        except Exception as e:
            default_logger.error(f"RAG检索器初始化失败: {e}")
            raise

    def _init_embedding_model(self) -> None:
        """初始化向量模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                EMBEDDING_MODEL,
                device=EMBEDDING_DEVICE
            )
            default_logger.info(f"向量模型加载成功: {EMBEDDING_MODEL}")
        except ImportError:
            default_logger.warning(
                "sentence_transformers未安装，使用模拟向量模式"
            )
            self._embedding_model = None
        except Exception as e:
            default_logger.error(f"向量模型加载失败: {e}")
            self._embedding_model = None

    def _encode_text(self, text: str) -> List[float]:
        """
        将文本编码为向量

        Args:
            text: 输入文本

        Returns:
            向量数组
        """
        if self._embedding_model is None:
            # 模拟向量（开发测试用）
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            # 生成512维模拟向量
            vector: List[float] = []
            for i in range(512):
                vector.append((hash_int + i) % 100 / 100.0)
            return vector

        # 使用真实向量模型
        embedding = self._embedding_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embedding.tolist()

    def retrieve(
        self,
        request: RetrievalRequest
    ) -> List[RetrievalResult]:
        """
        混合检索

        Args:
            request: 检索请求参数

        Returns:
            检索结果列表，按融合得分排序
        """
        if not request.query or not request.query.strip():
            return []

        try:
            # 执行向量检索
            vector_results = self._vector_search(request)

            # 执行关键词检索
            keyword_results = self._keyword_search(request)

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

        except Exception as e:
            default_logger.error(f"检索失败: {e}")
            return []

    def _vector_search(
        self,
        request: RetrievalRequest
    ) -> Dict[str, RetrievalResult]:
        """
        向量相似度检索

        Args:
            request: 检索请求参数

        Returns:
            字典: {chunk_id: RetrievalResult}
        """
        try:
            # 编码查询向量
            query_vector = self._encode_text(request.query)

            # 构建ES查询
            must_clauses = []
            filter_clauses = []

            # 元数据过滤
            if request.business_domain:
                filter_clauses.append({
                    "term": {"metadata.business_domain": request.business_domain}
                })

            if request.doc_type:
                filter_clauses.append({
                    "term": {"metadata.doc_type": request.doc_type}
                })

            # 有效期过滤
            if request.valid_date:
                valid_date_str = request.valid_date.strftime("%Y-%m-%d")
                filter_clauses.append({
                    "range": {
                        "metadata.valid_from": {"lte": valid_date_str}
                    }
                })
                filter_clauses.append({
                    "range": {
                        "metadata.valid_until": {"gte": valid_date_str}
                    }
                })

            # 组合查询条件
            if filter_clauses:
                bool_query = {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    "field": "embedding",
                                    "query_vector": query_vector,
                                    "k": request.vector_top_k,
                                    "num_candidates": request.vector_top_k * 2
                                }
                            }
                        ],
                        "filter": filter_clauses
                    }
                }
            else:
                bool_query = {
                    "knn": {
                        "field": "embedding",
                        "query_vector": query_vector,
                        "k": request.vector_top_k,
                        "num_candidates": request.vector_top_k * 2
                    }
                }

            # 执行查询
            response = self.es_client.search(
                index=self.index_name,
                body={
                    "query": bool_query,
                    "size": request.vector_top_k
                }
            )

            # 解析结果
            results: Dict[str, RetrievalResult] = {}
            for hit in response['hits']['hits']:
                chunk_id = hit['_source'].get('chunk_id', hit['_id'])
                results[chunk_id] = RetrievalResult(
                    content=hit['_source'].get('content', ''),
                    metadata=hit['_source'].get('metadata', {}),
                    score=hit['_score'],
                    vector_score=hit['_score'],
                    keyword_score=None,
                    doc_id=hit['_source'].get('doc_id'),
                    chunk_id=chunk_id
                )

            default_logger.info(
                f"向量检索返回 {len(results)} 条结果"
            )
            return results

        except Exception as e:
            default_logger.error(f"向量检索失败: {e}")
            return {}

    def _keyword_search(
        self,
        request: RetrievalRequest
    ) -> Dict[str, RetrievalResult]:
        """
        BM25关键词检索

        Args:
            request: 检索请求参数

        Returns:
            字典: {chunk_id: RetrievalResult}
        """
        try:
            # 构建ES查询
            must_clauses = [
                {
                    "match": {
                        "content": {
                            "query": request.query,
                            "operator": "or"
                        }
                    }
                }
            ]

            filter_clauses = []

            # 元数据过滤
            if request.business_domain:
                filter_clauses.append({
                    "term": {"metadata.business_domain": request.business_domain}
                })

            if request.doc_type:
                filter_clauses.append({
                    "term": {"metadata.doc_type": request.doc_type}
                })

            # 有效期过滤
            if request.valid_date:
                valid_date_str = request.valid_date.strftime("%Y-%m-%d")
                filter_clauses.append({
                    "range": {
                        "metadata.valid_from": {"lte": valid_date_str}
                    }
                })
                filter_clauses.append({
                    "range": {
                        "metadata.valid_until": {"gte": valid_date_str}
                    }
                })

            # 组合查询条件
            if filter_clauses:
                query = {
                    "bool": {
                        "must": must_clauses,
                        "filter": filter_clauses
                    }
                }
            else:
                query = {
                    "bool": {
                        "must": must_clauses
                    }
                }

            # 执行查询
            response = self.es_client.search(
                index=self.index_name,
                body={
                    "query": query,
                    "size": request.keyword_top_k
                }
            )

            # 解析结果
            results: Dict[str, RetrievalResult] = {}
            for hit in response['hits']['hits']:
                chunk_id = hit['_source'].get('chunk_id', hit['_id'])
                results[chunk_id] = RetrievalResult(
                    content=hit['_source'].get('content', ''),
                    metadata=hit['_source'].get('metadata', {}),
                    score=hit['_score'],
                    vector_score=None,
                    keyword_score=hit['_score'],
                    doc_id=hit['_source'].get('doc_id'),
                    chunk_id=chunk_id
                )

            default_logger.info(
                f"关键词检索返回 {len(results)} 条结果"
            )
            return results

        except Exception as e:
            default_logger.error(f"关键词检索失败: {e}")
            return {}

    def _fuse_results(
        self,
        vector_results: Dict[str, RetrievalResult],
        keyword_results: Dict[str, RetrievalResult],
        vector_weight: float,
        keyword_weight: float
    ) -> List[RetrievalResult]:
        """
        融合向量检索和关键词检索结果

        Args:
            vector_results: 向量检索结果
            keyword_results: 关键词检索结果
            vector_weight: 向量权重
            keyword_weight: 关键词权重

        Returns:
            融合后的结果列表
        """
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
                # 同时出现在两个检索中，进行加权融合
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
                # 只出现在向量检索中
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

    def create_index(self, index_name: Optional[str] = None) -> bool:
        """
        创建ES索引

        Args:
            index_name: 索引名称，默认使用配置中的索引名

        Returns:
            是否创建成功
        """
        try:
            target_index = index_name if index_name else self.index_name

            # 检查索引是否存在
            if self.es_client.indices.exists(index=target_index):
                default_logger.info(f"索引已存在: {target_index}")
                return True

            # 创建索引
            self.es_client.indices.create(
                index=target_index,
                body=INDEX_MAPPING
            )
            default_logger.info(f"索引创建成功: {target_index}")
            return True

        except Exception as e:
            default_logger.error(f"创建索引失败: {e}")
            return False

    def insert_documents(
        self,
        documents: List[Dict[str, Any]],
        index_name: Optional[str] = None
    ) -> int:
        """
        批量插入文档

        Args:
            documents: 文档列表，每项包含content、metadata等字段
            index_name: 索引名称

        Returns:
            成功插入的数量
        """
        try:
            target_index = index_name if index_name else self.index_name
            now = datetime.now().isoformat()

            actions = []
            for doc in documents:
                doc_id = doc.get('doc_id', '')
                content = doc.get('content', '')

                if not content:
                    continue

                # 生成向量
                embedding = self._encode_text(content)

                # 准备文档
                action = {
                    "_index": target_index,
                    "_id": doc_id if doc_id else None,
                    "_source": {
                        "doc_id": doc_id,
                        "chunk_id": doc.get('chunk_id', doc_id),
                        "content": content,
                        "embedding": embedding,
                        "metadata": doc.get('metadata', {}),
                        "created_at": now,
                        "updated_at": now
                    }
                }
                actions.append(action)

            if not actions:
                return 0

            # 批量插入
            from elasticsearch.helpers import bulk
            success_count, _ = bulk(self.es_client, actions)

            default_logger.info(
                f"成功插入 {success_count}/{len(actions)} 条文档"
            )
            return success_count

        except Exception as e:
            default_logger.error(f"插入文档失败: {e}")
            return 0

    def delete_index(self, index_name: Optional[str] = None) -> bool:
        """
        删除索引

        Args:
            index_name: 索引名称

        Returns:
            是否删除成功
        """
        try:
            target_index = index_name if index_name else self.index_name

            if not self.es_client.indices.exists(index=target_index):
                default_logger.info(f"索引不存在: {target_index}")
                return True

            self.es_client.indices.delete(index=target_index)
            default_logger.info(f"索引删除成功: {target_index}")
            return True

        except Exception as e:
            default_logger.error(f"删除索引失败: {e}")
            return False


# 创建全局检索器实例
rag_retriever = RAGRetriever()


# ============= 测试用例 =============
def test_rag_retriever() -> None:
    """
    测试RAG检索器
    """
    print("=" * 50)
    print("AI小益RAG检索模块测试")
    print("=" * 50)

    retriever = RAGRetriever()

    # 创建测试索引
    test_index = "test_finance_rag"
    print(f"\n1. 创建测试索引: {test_index}")
    if not retriever.create_index(test_index):
        print("   索引创建失败，跳过测试")
        return

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

    print("\n2. 插入测试文档...")
    insert_count = retriever.insert_documents(test_documents, test_index)
    print(f"   成功插入 {insert_count} 条文档")

    if insert_count == 0:
        print("   文档插入失败，跳过测试")
        retriever.delete_index(test_index)
        return

    # 等待索引刷新
    retriever.es_client.indices.refresh(index=test_index)

    # 测试混合检索
    print("\n3. 测试混合检索...")
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
    print("\n\n4. 测试元数据过滤...")
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
        print(f"    内容: {result.content[:80]}...")
        print(f"得分: {result.score:.4f}")
        print(f"类型: {result.metadata.get('doc_type')}")

    # 测试有效期过滤
    print("\n\n5. 测试有效期过滤...")
    request_validity = RetrievalRequest(
        query="净值型产品",
        business_domain="finance",
        valid_date=datetime(2024, 6, 15),
        vector_top_k=20,
        keyword_top_k=20,
        final_top_k=3
    )

    results_validity = retriever.retrieve(request_validity)

    print(f"\n   查询: {request_validity.query}")
    print(f"   查询日期: {request_validity.valid_date.strftime('%Y-%m-%d')}")
    print(f"   返回 {len(results_validity)} 条结果")
    print("-" * 50)

    for i, result in enumerate(results_validity, 1):
        print(f"\n   结果 {i}:")
        print(f"     内容: {result.content[:80]}...")
        print(f"     得分: {result.score:.4f}")

    # 清理测试索引
    print("\n\n6. 清理测试环境...")
    if retriever.delete_index(test_index):
        print("   测试索引已删除")

    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)


if __name__ == '__main__':
    test_rag_retriever()
