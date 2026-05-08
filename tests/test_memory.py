"""
Memory System 单元测试
测试短期记忆、中期记忆（Redis Mock）、长期记忆（ES Mock）
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List
from datetime import datetime

# 导入测试目标模块
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def mock_redis_client():
    """Mock Redis 客户端"""
    mock = MagicMock()
    mock.ping.return_value = True
    return mock


@pytest.fixture
def mock_es_client():
    """Mock Elasticsearch 客户端"""
    mock = MagicMock()
    mock.ping.return_value = True
    mock.indices.exists.return_value = False
    mock.index.return_value = {"result": "created"}
    mock.search.return_value = {
        "hits": {
            "hits": []
        }
    }
    return mock


@pytest.fixture
def memory_system_without_external():
    """创建不依赖外部服务的记忆系统"""
    with patch('agent.memory.redis') as mock_redis_module, \
         patch('agent.memory.HAS_CONFIG', False):

        # Mock Redis 导入失败，使用内存存储
        mock_redis_module.Redis.side_effect = ImportError("Redis not available")

        from agent.memory import MemorySystem
        memory = MemorySystem()
        return memory


@pytest.fixture
def memory_system_with_mock_external(mock_redis_client, mock_es_client):
    """创建带 Mock 外部服务的记忆系统"""
    with patch('agent.memory.redis') as mock_redis_module, \
         patch('agent.memory.config') as mock_config, \
         patch('agent.memory.HAS_CONFIG', True):

        # Mock Redis
        mock_redis_module.Redis.return_value = mock_redis_client

        # Mock ES
        mock_config.es_client = mock_es_client

        from agent.memory import MemorySystem
        memory = MemorySystem()

        # 重新设置 Mock 客户端（因为初始化中可能被覆盖）
        memory._redis_client = mock_redis_client
        memory._es_client = mock_es_client

        return memory, mock_redis_client, mock_es_client


# ==================== 测试短期记忆 ====================

class TestShortMemory:
    """测试短期记忆功能"""

    def test_add_episodic_memory(self, memory_system_without_external):
        """测试添加短期记忆"""
        memory = memory_system_without_external

        memory.add_episodic_memory(
            content="测试内容",
            metadata={"source": "test"},
            importance=0.8
        )

        assert len(memory._episodic_memory) == 1
        assert memory._episodic_memory[0].content == "测试内容"
        assert memory._episodic_memory[0].importance == 0.8

    def test_get_episodic_memory(self, memory_system_without_external):
        """测试获取短期记忆"""
        memory = memory_system_without_external

        # 添加多条记忆
        for i in range(5):
            memory.add_episodic_memory(f"内容{i}")

        # 获取最近的3条
        recent = memory.get_episodic_memory(limit=3)

        assert len(recent) == 3
        assert recent[0].content == "内容2"
        assert recent[2].content == "内容4"

    def test_episodic_memory_capacity_limit(self, memory_system_without_external):
        """测试短期记忆容量限制"""
        memory = memory_system_without_external

        # 添加超过容量的记忆（容量为10）
        for i in range(15):
            memory.add_episodic_memory(f"内容{i}")

        # 应该只保留最近的10条
        assert len(memory._episodic_memory) == 10
        assert memory._episodic_memory[0].content == "内容5"
        assert memory._episodic_memory[-1].content == "内容14"

    def test_clear_episodic_memory(self, memory_system_without_external):
        """测试清除短期记忆"""
        memory = memory_system_without_external

        # 添加记忆
        for i in range(5):
            memory.add_episodic_memory(f"内容{i}")

        assert len(memory._episodic_memory) == 5

        # 清除记忆
        memory.clear_episodic_memory()

        assert len(memory._episodic_memory) == 0

    def test_episodic_memory_with_metadata(self, memory_system_without_external):
        """测试带元数据的短期记忆"""
        memory = memory_system_without_external

        metadata = {
            "user_id": "user_123",
            "session_id": "session_456",
            "timestamp": "2024-01-01"
        }

        memory.add_episodic_memory(
            content="重要消息",
            metadata=metadata,
            importance=1.0
        )

        assert memory._episodic_memory[0].metadata == metadata
        assert memory._episodic_memory[0].importance == 1.0


# ==================== 测试中期记忆 ====================

class TestMediumMemory:
    """测试中期记忆功能"""

    def test_add_semantic_memory(self, memory_system_without_external):
        """测试添加语义记忆"""
        memory = memory_system_without_external

        memory.add_semantic_memory(
            key="user_123",
            content="用户偏好：喜欢简洁回答",
            tags=["preference", "user"]
        )

        assert "user_123" in memory._semantic_memory
        assert len(memory._semantic_memory["user_123"]) == 1
        assert memory._semantic_memory["user_123"][0].content == "用户偏好：喜欢简洁回答"

    def test_get_semantic_memory(self, memory_system_without_external):
        """测试获取语义记忆"""
        memory = memory_system_without_external

        # 添加多条语义记忆
        for i in range(10):
            memory.add_semantic_memory(
                key="user_123",
                content=f"偏好{i}",
                tags=["preference"]
            )

        # 获取最近的3条
        recent = memory.get_semantic_memory(key="user_123", limit=3)

        assert len(recent) == 3
        assert recent[0].content == "偏好7"
        assert recent[2].content == "偏好9"

    def test_semantic_memory_capacity_limit(self, memory_system_without_external):
        """测试语义记忆容量限制"""
        memory = memory_system_without_external

        # 添加超过容量的记忆（容量为100）
        for i in range(105):
            memory.add_semantic_memory(
                key="user_123",
                content=f"内容{i}"
            )

        # 应该只保留最近的100条
        assert len(memory._semantic_memory["user_123"]) == 100
        assert memory._semantic_memory["user_123"][0].content == "内容5"

    def test_semantic_memory_multiple_keys(self, memory_system_without_external):
        """测试多个键的语义记忆"""
        memory = memory_system_without_external

        # 为不同用户添加记忆
        memory.add_semantic_memory(key="user_1", content="用户1偏好")
        memory.add_semantic_memory(key="user_2", content="用户2偏好")
        memory.add_semantic_memory(key="user_3", content="用户3偏好")

        assert len(memory._semantic_memory) == 3
        assert "user_1" in memory._semantic_memory
        assert "user_2" in memory._semantic_memory
        assert "user_3" in memory._semantic_memory

    def test_clear_semantic_memory_specific_key(self, memory_system_without_external):
        """测试清除特定键的语义记忆"""
        memory = memory_system_without_external

        memory.add_semantic_memory(key="user_1", content="偏好1")
        memory.add_semantic_memory(key="user_2", content="偏好2")

        assert len(memory._semantic_memory) == 2

        # 清除特定键
        memory.clear_semantic_memory(key="user_1")

        assert len(memory._semantic_memory) == 1
        assert "user_1" not in memory._semantic_memory
        assert "user_2" in memory._semantic_memory

    def test_clear_all_semantic_memory(self, memory_system_without_external):
        """测试清除所有语义记忆"""
        memory = memory_system_without_external

        memory.add_semantic_memory(key="user_1", content="偏好1")
        memory.add_semantic_memory(key="user_2", content="偏好2")

        assert len(memory._semantic_memory) == 2

        # 清除所有
        memory.clear_semantic_memory(key=None)

        assert len(memory._semantic_memory) == 0

    def test_semantic_memory_with_tags(self, memory_system_without_external):
        """测试带标签的语义记忆"""
        memory = memory_system_without_external

        memory.add_semantic_memory(
            key="user_123",
            content="喜欢金融产品",
            tags=["finance", "preference"]
        )

        assert memory._semantic_memory["user_123"][0].tags == ["finance", "preference"]


# ==================== 测试长期记忆 ====================

class TestLongMemory:
    """测试长期记忆功能"""

    def test_add_long_term_memory(self, memory_system_without_external):
        """测试添加长期记忆"""
        memory = memory_system_without_external

        memory.add_long_term_memory(
            content="重要知识：固收类产品风险较低",
            metadata={"category": "product_knowledge"},
            importance=1.0,
            tags=["finance", "product"]
        )

        assert len(memory._long_term_memory) == 1
        assert memory._long_term_memory[0].content == "重要知识：固收类产品风险较低"
        assert memory._long_term_memory[0].importance == 1.0

    def test_get_long_term_memory_without_query(self, memory_system_without_external):
        """测试不带查询获取长期记忆"""
        memory = memory_system_without_external

        # 添加多条长期记忆
        for i in range(10):
            memory.add_long_term_memory(f"知识{i}")

        # 获取最近的3条
        recent = memory.get_long_term_memory(limit=3)

        assert len(recent) == 3
        assert recent[0].content == "知识7"
        assert recent[2].content == "知识9"

    def test_long_term_memory_capacity_limit(self, memory_system_without_external):
        """测试长期记忆容量限制"""
        memory = memory_system_without_external

        # 添加超过容量的记忆（容量为1000）
        for i in range(1005):
            memory.add_long_term_memory(f"内容{i}")

        # 应该只保留最近的1000条
        assert len(memory._long_term_memory) == 1000
        assert memory._long_term_memory[0].content == "内容5"

    def test_long_term_memory_with_tags(self, memory_system_without_external):
        """测试带标签的长期记忆"""
        memory = memory_system_without_external

        memory.add_long_term_memory(
            content="知识内容",
            tags=["knowledge", "important"]
        )

        assert memory._long_term_memory[0].tags == ["knowledge", "important"]


# ==================== 测试 Elasticsearch 集成 ====================

class TestElasticsearchIntegration:
    """测试 Elasticsearch 集成"""

    def test_sync_to_es_success(self, memory_system_with_mock_external):
        """测试成功同步到 ES"""
        memory, mock_redis, mock_es = memory_system_with_mock_external

        # 添加长期记忆（会触发同步）
        memory.add_long_term_memory(
            content="测试知识",
            metadata={"test": True}
        )

        # 验证 ES 调用
        mock_es.index.assert_called_once()

    def test_sync_to_es_with_existing_index(self, memory_system_with_mock_external):
        """测试同步到已存在的 ES 索引"""
        memory, mock_redis, mock_es = memory_system_with_mock_external

        # 模拟索引已存在
        mock_es.indices.exists.return_value = True

        memory.add_long_term_memory("测试内容")

        # 验证没有创建索引调用
        mock_es.indices.create.assert_not_called()

    def test_search_long_term_from_es(self, memory_system_with_mock_external):
        """测试从 ES 检索长期记忆"""
        memory, mock_redis, mock_es = memory_system_with_mock_external

        # 模拟 ES 搜索结果
        mock_es.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_source": {
                            "content": "搜索结果1",
                            "timestamp": "2024-01-01T00:00:00",
                            "metadata": {},
                            "importance": 1.0,
                            "tags": []
                        }
                    },
                    {
                        "_source": {
                            "content": "搜索结果2",
                            "timestamp": "2024-01-02T00:00:00",
                            "metadata": {},
                            "importance": 0.8,
                            "tags": []
                        }
                    }
                ]
            }
        }

        results = memory.get_long_term_memory(query="测试查询", limit=5)

        assert len(results) == 2
        assert results[0].content == "搜索结果1"
        assert results[1].content == "搜索结果2"

    def test_search_with_es_disconnected(self, memory_system_with_mock_external):
        """测试 ES 断开连接时使用本地存储"""
        memory, mock_redis, mock_es = memory_system_with_mock_external

        # 先添加一些本地记忆
        memory.add_long_term_memory("本地知识1")
        memory.add_long_term_memory("本地知识2")

        # 模拟 ES 连接失败
        mock_es.ping.return_value = False

        results = memory.get_long_term_memory(query="测试", limit=5)

        # 应该返回本地存储的结果
        assert len(results) >= 2

    def test_sync_to_es_failure(self, memory_system_with_mock_external):
        """测试 ES 同步失败的情况"""
        memory, mock_redis, mock_es = memory_system_with_mock_external

        # 模拟 ES 同步失败
        mock_es.index.side_effect = Exception("ES 连接失败")

        # 添加长期记忆（同步失败不应该影响本地存储）
        memory.add_long_term_memory("测试内容")

        # 本地存储应该成功
        assert len(memory._long_term_memory) == 1


# ==================== 测试 Agent 记忆 ====================

class TestAgentMemory:
    """测试 Agent 记忆功能"""

    def test_create_agent_memory(self, memory_system_without_external):
        """测试创建 Agent 记忆状态"""
        memory = memory_system_without_external

        # 添加一些记忆数据
        memory.add_episodic_memory("用户查询1")
        memory.add_episodic_memory("用户查询2")
        memory.add_semantic_memory(key="user_123", content="用户偏好")

        agent_memory = memory.create_agent_memory(
            session_id="session_456",
            user_id="user_123",
            current_query="当前查询",
            context="对话上下文"
        )

        assert agent_memory.session_id == "session_456"
        assert agent_memory.user_id == "user_123"
        assert agent_memory.current_query == "当前查询"
        assert agent_memory.context == "对话上下文"
        assert len(agent_memory.conversation_history) >= 2

    def test_get_user_preferences(self, memory_system_without_external):
        """测试获取用户偏好"""
        memory = memory_system_without_external

        # 添加用户偏好
        memory.add_semantic_memory(key="user_123", content="偏好1")
        memory.add_semantic_memory(key="user_123", content="偏好2")

        preferences = memory._get_user_preferences("user_123")

        assert "pref_0" in preferences
        assert "pref_1" in preferences
        assert preferences["pref_0"] == "偏好1"

    def test_get_user_preferences_no_user(self, memory_system_without_external):
        """测试不传用户ID时的偏好获取"""
        memory = memory_system_without_external

        preferences = memory._get_user_preferences(None)

        assert preferences == {}

    def test_get_conversation_history(self, memory_system_without_external):
        """测试获取对话历史"""
        memory = memory_system_without_external

        # 添加对话历史
        memory.add_episodic_memory("用户消息1", metadata={"role": "user"})
        memory.add_episodic_memory("系统回复1", metadata={"role": "assistant"})

        history = memory._get_conversation_history("session_123")

        assert len(history) == 2
        assert history[0]["content"] == "用户消息1"
        assert history[1]["content"] == "系统回复1"


# ==================== 测试导出功能 ====================

class TestMemoryExport:
    """测试记忆导出功能"""

    def test_export_memory(self, memory_system_without_external):
        """测试导出所有记忆"""
        memory = memory_system_without_external

        # 添加各种类型的记忆
        memory.add_episodic_memory("短期记忆")
        memory.add_semantic_memory(key="user_1", content="语义记忆")
        memory.add_long_term_memory("长期记忆")

        exported = memory.export_memory()

        assert "episodic" in exported
        assert "semantic" in exported
        assert "long_term" in exported
        assert "stats" in exported

        assert len(exported["episodic"]) == 1
        assert len(exported["semantic"]) == 1
        assert len(exported["long_term"]) == 1

        assert exported["stats"]["episodic_size"] == 1
        assert exported["stats"]["semantic_keys"] == 1
        assert exported["stats"]["long_term_size"] == 1

    def test_export_empty_memory(self, memory_system_without_external):
        """测试导出空记忆"""
        memory = memory_system_without_external

        exported = memory.export_memory()

        assert exported["stats"]["episodic_size"] == 0
        assert exported["stats"]["semantic_keys"] == 0
        assert exported["stats"]["long_term_size"] == 0


# ==================== 测试初始化 ====================

class TestMemoryInitialization:
    """测试记忆系统初始化"""

    def test_initialization_without_external_services(self, memory_system_without_external):
        """测试不依赖外部服务的初始化"""
        memory = memory_system_without_external

        assert memory._initialized is True
        assert memory._redis_client is None
        assert memory._es_client is None

    def test_initialization_with_mock_es(self, memory_system_with_mock_external):
        """测试带 Mock ES 的初始化"""
        memory, mock_redis, mock_es = memory_system_with_mock_external

        assert memory._initialized is True
        assert memory._es_client is not None
        assert mock_es.ping.called


# ==================== 测试异常处理 ====================

class TestErrorHandling:
    """测试异常处理"""

    def test_redis_connection_failure(self):
        """测试 Redis 连接失败处理"""
        with patch('agent.memory.redis') as mock_redis_module, \
             patch('agent.memory.HAS_CONFIG', False):

            # Mock Redis 连接失败
            mock_redis_module.Redis.side_effect.return_value = None

            from agent.memory import MemorySystem
            memory = MemorySystem()

            # 应该使用内存存储作为回退
            assert memory._redis_client is None

    def test_es_connection_failure(self):
        """测试 ES 连接失败处理"""
        with patch('agent.memory.redis') as mock_redis_module, \
             patch('agent.memory.config') as mock_config, \
             patch('agent.memory.HAS_CONFIG', True):

            mock_redis_module.Redis.side_effect = ImportError("Redis not available")
            mock_config.es_client = None

            from agent.memory import MemorySystem
            memory = MemorySystem()

            # 应该能正常初始化
            assert memory._initialized is True
            assert memory._es_client is None

    def test_add_memory_with_none_content(self, memory_system_without_external):
        """测试添加空内容的记忆"""
        memory = memory_system_without_external

        # 不应该抛出异常
        memory.add_episodic_memory("")
        memory.add_semantic_memory(key="test", content="")
        memory.add_long_term_memory("")

        # 空内容也应该被存储
        assert len(memory._episodic_memory) == 1

    def test_get_memory_with_invalid_key(self, memory_system_without_external):
        """测试获取不存在的键的记忆"""
        memory = memory_system_without_external

        results = memory.get_semantic_memory(key="nonexistent_key")

        assert results == []
