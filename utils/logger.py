"""
日志工具模块
统一的日志配置
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """
    日志工具类
    提供统一的日志配置
    """

    _loggers: dict[str, logging.Logger] = {}

    @classmethod
    def get_logger(
        cls,
        name: str,
        log_file: Optional[str] = None,
        log_level: int = logging.INFO,
        console_output: bool = True
    ) -> logging.Logger:
        """
        获取或创建日志记录器

        Args:
            name: 日志记录器名称
            log_file: 日志文件路径
            log_level: 日志级别
            console_output: 是否输出到控制台

        Returns:
            日志记录器实例
        """
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger


# 创建默认日志记录器
default_logger = Logger.get_logger('ai_xiaoyi')


def get_logger(name: str) -> logging.Logger:
    """
    快速获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器实例
    """
    return Logger.get_logger(name)
