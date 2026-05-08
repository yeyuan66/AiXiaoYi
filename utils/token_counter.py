"""
Token计数工具模块
使用tiktoken适配GLM模型进行Token计数
"""

import tiktoken
from typing import Optional


class TokenCounter:
    """
    Token计数器
    使用tiktoken库对文本进行Token计数
    """

    def __init__(self, encoding_name: Optional[str] = None) -> None:
        """
        初始化Token计数器

        Args:
            encoding_name: tiktoken编码名称，默认使用cl100k_base（GLM兼容）
        """
        if encoding_name is None:
            encoding_name = "cl100k_base"
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """
        计算文本的Token数量

        Args:
            text: 待计数的文本

        Returns:
            Token数量
        """
        if not text:
            return 0
        return len(self.encoding.encode(text))

    def count_messages_tokens(
        self,
        messages: list[dict[str, str]]
    ) -> int:
        """
        计算消息列表的Token数量
        兼容langchain的消息格式

        Args:
            messages: 消息列表，格式为[{"role": "...", "content": "..."}, ...]

        Returns:
            总Token数量
        """
        if not messages:
            return 0

        total_tokens = 0
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            # 每个消息有固定的Token开销
            total_tokens += 4  # role和content的开销
            total_tokens += self.count_tokens(role)
            total_tokens += self.count_tokens(content)

        # 整体有额外的Token开销
        total_tokens += 2
        return total_tokens

    def truncate_to_tokens(
        self,
        text: str,
        max_tokens: int
    ) -> str:
        """
        截断文本到指定的Token数量

        Args:
            text: 原始文本
            max_tokens: 最大Token数量

        Returns:
            �截断后的文本
        """
        if not text:
            return ""

        tokens = self.encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text

        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)

    def estimate_tokens_from_chars(
        self,
        text: str,
        factor: float = 0.3
    ) -> int:
        """
        基于字符数快速估算Token数量
        这是一个粗略估算，不推荐用于精确计算

        Args:
            text: 待估算的文本
            factor: 字符到Token的转换因子，默认0.3（中文字符通常0.3个Token）

        Returns:
            估算的Token数量
        """
        if not text:
            return 0
        return int(len(text) * factor)


# 创建全局计数器实例
default_counter = TokenCounter()
