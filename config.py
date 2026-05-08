"""
AI小益全局配置模块
负责读取环境变量并初始化各类客户端
"""

import os
from dataclasses import dataclass
from typing import Optional, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from elasticsearch import Elasticsearch


# 全局常量定义
MAX_RETRY_COUNT: int = 2
MAX_EXECUTE_STEPS: int = 3
CONTEXT_WINDOW_LIMIT: int = 116000
OUTPUT_RESERVED_TOKEN: int = 12000


class Config:
    """
    配置单例类
    读取.env文件中的所有配置，并提供各类客户端的初始化
    """

    _instance: Optional['Config'] = None

    def __new__(cls) -> 'Config':
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化配置，从环境变量读取"""
        if not hasattr(self, '_initialized'):
            load_dotenv()

            # 客户端实例（延迟初始化）
            self._llm_client: Optional[ChatOpenAI] = None
            self._es_client: Optional[Elasticsearch] = None

            # 大模型配置
            self.openai_api_base = os.getenv(
                'OPENAI_API_BASE',
                'https://ark.cn-beijing.volces.com/api/v3'
            )
            self.openai_api_key = os.getenv('OPENAI_API_KEY', '')
            self.llm_model = os.getenv('LLM_MODEL', 'glm-4-7b-32k')
            self.llm_temperature = float(os.getenv('LLM_TEMPERATURE', '0.7'))
            self.llm_max_tokens = int(os.getenv('LLM_MAX_TOKENS', '4000'))

            # Elasticsearch配置
            self.es_host = os.getenv('ES_HOST', 'http://localhost:9200')
            self.es_user = os.getenv('ES_USER')
            self.es_password = os.getenv('ES_PASSWORD')
            self.es_index = os.getenv('ES_INDEX', 'finance_rag_knowledge_v1')

            # FastText意图模型配置
            self.fasttext_model_path = os.getenv(
                'FASTTEXT_MODEL_PATH',
                './models/intent_model.bin'
            )

            self._initialized = True

    @property
    def llm_client(self) -> ChatOpenAI:
        """获取或初始化大模型客户端"""
        if self._llm_client is None:
            self._llm_client = ChatOpenAI(
                model=self.llm_model,
                base_url=self.openai_api_base,
                api_key=self.openai_api_key,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )
        return self._llm_client

    @property
    def es_client(self) -> Elasticsearch:
        """获取或初始化Elasticsearch客户端"""
        if self._es_client is None:
            if self.es_user and self.es_password:
                self._es_client = Elasticsearch(
                    [self.es_host],
                    basic_auth=(self.es_user, self.es_password),
                    timeout=30,
                    max_retries=MAX_RETRY_COUNT,
                    verify_certs=False
                )
            else:
                self._es_client = Elasticsearch(
                    [self.es_host],
                    timeout=30,
                    max_retries=MAX_RETRY_COUNT,
                    verify_certs=False
                )
        return self._es_client

    def get_llm(self, **kwargs: Any) -> ChatOpenAI:
        """
        获取自定义配置的大模型实例
        允许覆盖默认配置参数
        """
        return ChatOpenAI(
            model=kwargs.get('model', self.llm_model),
            base_url=kwargs.get('base_url', self.openai_api_base),
            api_key=kwargs.get('api_key', self.openai_api_key),
            temperature=kwargs.get('temperature', self.llm_temperature),
            max_tokens=kwargs.get('max_tokens', self.llm_max_tokens),
            **{k: v for k, v in kwargs.items() if k not in [
                'model', 'base_url', 'api_key', 'temperature', 'max_tokens'
            ]}
        )


# 创建全局配置实例
config = Config()
