"""
Pytest 配置和共享 fixtures
提供全局测试配置和 mock 设置
"""

import pytest
import sys
import os
from unittest.mock import Mock, MagicMock, patch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==================== 全局配置 ====================

def pytest_configure(config):
    """pytest 配置钩子"""
    config.addinivalue_line(
        "markers", "slow: 标记慢速测试"
    )
    config.addinivalue_line(
        "markers", "integration: 标记集成测试"
    )
    config.addinivalue_line(
        "markers", "unit: 标记单元测试"
    )


# ==================== 全局 Mock 设置 ====================

@pytest.fixture(scope="session")
def mock_config():
    """全局 Mock 配置模块"""
    with patch('agent.core.HAS_CONFIG', True), \
         patch('agent.memory.HAS_CONFIG', False):

        # 创建 Mock 配置
        class MockConfig:
            def __init__(self):
                self.llm_client = MagicMock()
                self.llm_client.invoke.return_value = MagicMock(content="模拟LLM响应")
                self.es_client = None

        return MockConfig()


# ==================== 测试环境清理 ====================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """每个测试后自动清理"""
    yield
    # 清理逻辑
    import gc
    gc.collect()


# ==================== 跳过外部服务测试 ====================

def pytest_collection_modifyitems(config, items):
    """修改测试收集，标记需要外部服务的测试"""
    for item in items:
        # 如果测试名称包含 external 或 redis 或 es，标记为 integration
        if any(keyword in item.name.lower()
               for keyword in ["external", "redis", "elasticsearch"]):
            item.add_marker(pytest.mark.integration)
