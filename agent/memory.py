"""
三级记忆系统模块
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum

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
    default_logger = logging.getLogger('memory_system')


# 记忆类型枚举
class MemoryType(Enum):
    """记忆类型"""
    EPHEMERAL = "episodic"      # 短期记忆（当前会话）
    SEMANTIC = "semantic"        # 语义记忆（用户偏好）
    LONG_TERM = "long_term"        # 长期记忆（全局知识）


  

@dataclass
class MemoryItem:
    """记忆项"""
    content: str                  # 记忆内容
    timestamp: float              # 时间戳
    metadata: Optional[Dict[str, Any]] = None  # 元数据
    importance: float = 1.0       # 重要程度（0.0-1.0）
    tags: Optional[List[str]] = None    # 标签


@dataclass
class AgentMemory:
    """Agent记忆状态"""
    session_id: str              # 会话ID
    current_query: str          # 当前查询
    preferences: Dict[str, Any]     # 用户偏好
    conversation_history: List[Dict[str, Any]]  # 对话历史
    user_id: Optional[str] = None       # 用户ID
    context: Optional[str] = None        # 上下文


class MemorySystem:
    """
    三级记忆系统
    1. 短期记忆（当前会话）：使用BufferMemory
    2. 中期记忆（用户偏好）：Redis存储
    3. 长期记忆（全局知识）：Elasticsearch存储
    """

    def __init__(self):
        """初始化记忆系统"""
        self._episodic_memory: List[MemoryItem] = []
        self._semantic_memory: Dict[str, List[MemoryItem]] = {}
        self._long_term_memory: List[MemoryItem] = []

        self._max_episodic_size = 10        # 短期记忆最大容量
        self._max_semantic_size = 100        # 语义记忆最大容量
        self._max_long_term_size = 1000       # 长期记忆最大容量

        self._initialized = False

        self._initialize()

    def _initialize(self) -> None:
        """初始化记忆系统"""
        try:
            # 尝试连接Redis（中期记忆）
            self._init_redis()
            # 尝试连接ES（长期记忆）
            self._init_es()

            self._initialized = True
            default_logger.info("记忆系统初始化成功")

        except Exception as e:
            default_logger.error(f"记忆系统初始化失败: {e}")
            # 使用内存存储作为回退
            self._initialized = True

    def _init_redis(self) -> None:
        """初始化Redis连接（中期记忆）"""
        try:
            import redis
            self._redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True
            )
            self._redis_client.ping()
            default_logger.info("Redis连接成功（中期记忆）")
        except Exception as e:
            default_logger.warning(f"Redis连接失败: {e}，使用内存存储")
            self._redis_client = None

    def _init_es(self) -> None:
        """初始化ES连接（长期记忆）"""
        try:
            if HAS_CONFIG:
                self._es_client = config.es_client
                self._es_client.ping()
                default_logger.info("ES连接成功（长期记忆）")
            else:
                self._es_client = None
                default_logger.warning("ES客户端未配置")
        except Exception as e:
            default_logger.warning(f"ES连接失败: {e}")
            self._es_client = None

    def add_episodic_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0
    ) -> None:
        """
        添加短期记忆

        Args:
            content: 记忆内容
            metadata: 元数据
            importance: 重要程度
        """
        memory_item = MemoryItem(
            content=content,
            timestamp=datetime.now().timestamp(),
            metadata=metadata,
            importance=importance
        )

        self._episodic_memory.append(memory_item)

        # 限制容量
        if len(self._episodic_memory) > self._max_episodic_size:
            self._episodic_memory.pop(0)

        default_logger.debug(f"添加短期记忆: {content[:50]}...")

    def add_semantic_memory(
        self,
        key: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        添加中期记忆（语义记忆）

        Args:
            key: 记忆键（如用户ID、查询类型等）
            content: 记忆内容
            metadata: 元数据
            tags: 标签
        """
        memory_item = MemoryItem(
            content=content,
            timestamp=datetime.now().timestamp(),
            metadata=metadata,
            tags=tags
        )

        if key not in self._semantic_memory:
            self._semantic_memory[key] = []

        self._semantic_memory[key].append(memory_item)

        # 限制容量
        if len(self._semantic_memory[key]) > self._max_semantic_size:
            self._semantic_memory[key].pop(0)

        default_logger.debug(f"添加语义记忆: {key} - {content[:50]}...")

    def add_long_term_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        tags: Optional[List[str]] = None
    ) -> None:
        """
        添加长期记忆

        Args:
            content: 记忆内容
            metadata: 元数据
            importance: 重要程度
            tags: 标签
        """
        memory_item = MemoryItem(
            content=content,
            timestamp=datetime.now().timestamp(),
            metadata=metadata,
            importance=importance,
            tags=tags
        )

        self._long_term_memory.append(memory_item)

        # 限制容量
        if len(self._long_term_memory) > self._max_long_term_size:
            self._long_term_memory.pop(0)

        # 同步到ES
        self._sync_to_es(memory_item)

        default_logger.debug(f"添加长期记忆: {content[:50]}...")

    def _sync_to_es(self, item: MemoryItem) -> None:
        """同步到ES（长期记忆）"""
        try:
            if self._es_client is None:
                return

            # 检查ES连接
            if not self._es_client.ping():
                default_logger.warning("ES连接断开，跳过同步")
                return

            # 准备文档
            doc = {
                "content": item.content,
                "timestamp": datetime.fromtimestamp(item.timestamp).isoformat(),
                "metadata": item.metadata or {},
                "importance": item.importance,
                "tags": item.tags or []
            }

            # 插入到ES
            index_name = "agent_long_term_memory"
            if not self._es_client.indices.exists(index=index_name):
                self._es_client.indices.create(index=index_name, ignore=400)

            self._es_client.index(
                index=index_name,
                document=doc
            )

            default_logger.debug(f"同步到ES成功")

        except Exception as e:
            default_logger.warning(f"同步到ES失败: {e}")

    def get_episodic_memory(self, limit: int = 5) -> List[MemoryItem]:
        """
        获取短期记忆

        Args:
            limit: 返回数量限制

        Returns:
            记忆项列表
        """
        return self._episodic_memory[-limit:]

    def get_semantic_memory(
        self,
        key: str,
        limit: int = 10
    ) -> List[MemoryItem]:
        """
        获取语义记忆

        Args:
            key: 记忆键
            limit: 返回数量限制

        Returns:
            记忆项列表
        """
        if key not in self._semantic_memory:
            return []

        return self._semantic_memory[key][-limit:]

    def get_long_term_memory(
        self,
        query: Optional[str] = None,
        limit: int = 5
    ) -> List[MemoryItem]:
        """
        获取长期记忆

        Args:
            query: 查询文本（可选）
            limit: 返回数量限制

        Returns:
            记忆项列表
        """
        if query is None:
            return self._long_term_memory[-limit:]

        # 从ES检索相关记忆
        return self._search_long_term_memory(query, limit)

    def _search_long_term_memory(
        self,
        query: str,
        limit: int
    ) -> List[MemoryItem]:
        """从ES检索长期记忆"""
        try:
            if self._es_client is None:
                return self._long_term_memory[-limit:]

            # 检查ES连接
            if not self._es_client.ping():
                return self._long_term_memory[-limit:]

            # 搜索ES
            index_name = "agent_long_term_memory"
            if not self._es_client.indices.exists(index=index_name):
                return self._long_term_memory[-limit:]

            search_body = {
                "query": {
                    "match": {
                        "content": query
                    }
                },
                "size": limit,
                "sort": [
                    {"timestamp": {"order": "desc"}}
                ]
            }

            response = self._es_client.search(
                index=index_name,
                body=search_body
            )

            # 解析结果
            items = []
            for hit in response['hits']['hits']:
                source = hit['_source']
                items.append(MemoryItem(
                    content=source.get('content', ''),
                    timestamp=datetime.fromisoformat(
                        source.get('timestamp', '')
                    ).timestamp(),
                    metadata=source.get('metadata', {}),
                    importance=source.get('importance', 1.0),
                    tags=source.get('tags', [])
                ))

            return items

        except Exception as e:
            default_logger.warning(f"ES检索失败: {e}，使用内存存储")
            return self._long_term_memory[-limit:]

    def clear_episodic_memory(self) -> None:
        """清除短期记忆"""
        self._episodic_memory.clear()
        default_logger.info("短期记忆已清除")

    def clear_semantic_memory(self, key: Optional[str] = None) -> None:
        """
        清除语义记忆

        Args:
            key: 记忆键，None表示清除全部
        """
        if key is None:
            self._semantic_memory.clear()
            default_logger.info("所有语义记忆已清除")
        elif key in self._semantic_memory:
            del self._semantic_memory[key]
            default_logger.info(f"语义记忆已清除: {key}")

    def create_agent_memory(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        current_query: str = "",
        context: Optional[str] = None
    ) -> AgentMemory:
        """
        创建Agent记忆状态

        Args:
            session_id: 会话ID
            user_id: 用户ID
            current_query: 当前查询
            context: 上下文

        Returns:
            Agent记忆状态
        """
        return AgentMemory(
            session_id=session_id,
            user_id=user_id,
            current_query=current_query,
            context=context,
            preferences=self._get_user_preferences(user_id),
            conversation_history=self._get_conversation_history(session_id)
        )

    def _get_user_preferences(self, user_id: Optional[str]) -> Dict[str, Any]:
        """获取用户偏好（从中期记忆）"""
        if user_id is None:
            return {}

        preferences = self.get_semantic_memory(user_id, limit=10)

        # 转换为字典格式
        return {
            f"pref_{i}": item.content
            for i, item in enumerate(preferences)
        }

    def _get_conversation_history(
        self,
        session_id: str
    ) -> List[Dict[str, Any]]:
        """获取对话历史（从短期记忆）"""
        history = []

        for item in self.get_episodic_memory():
            history.append({
                "content": item.content,
                "timestamp": item.timestamp,
                "metadata": item.metadata
            })

        return history

    def export_memory(self) -> Dict[str, Any]:
        """导出所有记忆"""
        return {
            "episodic": [asdict(item) for item in self._episodic_memory],
            "semantic": {
                key: [asdict(item) for item in items]
                for key, items in self._semantic_memory.items()
            },
            "long_term": [asdict(item) for item in self._long_term_memory],
            "stats": {
                "episodic_size": len(self._episodic_memory),
                "semantic_keys": len(self._semantic_memory),
                "long_term_size": len(self._long_term_memory)
            }
        }


# 创建全局记忆系统实例
memory_system = MemorySystem()
