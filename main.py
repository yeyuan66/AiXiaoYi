"""
AI小益 - 招商银行收益升级中台智能化运营助手
程序入口
"""

import sys
import uuid
from datetime import datetime
from typing import Generator

try:
    from config import config
    from utils import default_logger
    from agent.core import AgentCore, AgentState
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    default_logger = logging.getLogger('main')


def main() -> None:
    """
    主函数
    """
    default_logger.info("=" * 60)
    default_logger.info("AI小益 - 招商银行收益升级中台智能化运营助手")
    default_logger.info("=" * 60)

    try:
        # 显示配置信息
        default_logger.info("系统配置:")
        if HAS_CONFIG:
            default_logger.info(f"  大模型: {config.llm_model}")
            default_logger.info(f"  ES连接: {config.es_host}")
            default_logger.info(f"  ES索引: {config.es_index}")
        else:
            default_logger.warning("  配置模块未加载")

        # 初始化意图识别器
        default_logger.info("正在初始化意图识别器...")
        try:
            from intent.recognizer import intent_recognizer
            default_logger.info("意图识别器初始化成功")
        except Exception as e:
            default_logger.error(f"意图识别器初始化失败: {e}")

        # 初始化RAG检索器
        default_logger.info("正在初始化RAG检索器...")
        try:
            from rag.retriever import rag_retriever
            default_logger.info("RAG检索器初始化成功")
        except Exception as e:
            default_logger.error(f"RAG检索器初始化失败: {e}")

        # 初始化Agent Core
        default_logger.info("正在初始化Agent Core...")
        try:
            from agent.core import agent_core
            default_logger.info("Agent Core初始化成功")
        except Exception as e:
            default_logger.error(f"Agent Core初始化失败: {e}")

        # 启动服务
        default_logger.info("正在启动服务...")
        try:
            from mcp.scheduler import skill_scheduler
            default_logger.info("服务启动成功")
        except Exception as e:
            default_logger.error(f"服务启动失败: {e}")

        default_logger.info("AI小益启动成功")
        default_logger.info("\n使用方法:")
        default_logger.info("  1. 交互式模式")
        default_logger.info("  2. 流式查询模式")

        # 选择启动模式
        mode = input("\n请选择启动模式 (1-2，默认1): ").strip() or "1"

        if mode == "1":
            interactive_mode()
        elif mode == "2":
            stream_query_mode()
        else:
            default_logger.warning(f"无效模式: {mode}，进入交互模式")
            interactive_mode()

    except KeyboardInterrupt:
        default_logger.info("\n用户中断，程序退出")
    except Exception as e:
        default_logger.error(f"启动失败: {e}", exc_info=True)
        raise


def interactive_mode() -> None:
    """
    交互式模式
    """
    default_logger.info("\n进入交互式模式...")
    default_logger.info("输入 'quit' 或 'exit' 退出\n")

    try:
        from agent.core import agent_core

        while True:
            try:
                # 获取用户输入
                user_input = input("AI小益> ").strip()

                # 检查退出命令
                if user_input.lower() in ['quit', 'exit']:
                    default_logger.info("退出交互模式")
                    break

                if not user_input:
                    continue

                # 显示处理中提示
                print("正在处理...\n")

                # 调用Agent处理
                result = process_query(user_input, agent_core)

                # 显示结果
                print(result['final_answer'])
                print("-" * 60)

            except KeyboardInterrupt:
                default_logger.info("用户中断")
                break
            except Exception as e:
                default_logger.error(f"处理失败: {e}")
                print(f"抱歉，处理您的请求时遇到了问题：{e}")
                print("请重试或稍后再试。\n")

    except Exception as e:
        default_logger.error(f"交互模式失败: {e}")


def stream_query_mode() -> None:
    """
    流式查询模式
    """
    default_logger.info("\n进入流式查询模式...")

    try:
        from agent.core import agent_core

        while True:
            try:
                # 获取用户输入
                user_input = input("AI小益> ").strip()

                # 检查退出命令
                if user_input.lower() in ['quit', 'exit']:
                    default_logger.info("退出流式查询模式")
                    break

                if not user_input:
                    continue

                # 流式处理
                print("AI小益: ", end='', flush=True)

                for token in stream_process_query(user_input, agent_core):
                    print(token, end='', flush=True)

                print("\n")

            except KeyboardInterrupt:
                default_logger.info("用户中断")
                break
            except Exception as e:
                default_logger.error(f"流式处理失败: {e}")
                print(f"抱歉，处理您的请求时遇到了问题：{e}\n")

    except Exception as e:
        default_logger.error(f"流式查询模式失败: {e}")


def process_query(query: str, agent: AgentCore) -> dict:
    """
    处理查询

    Args:
        query: 用户查询
        agent: Agent Core实例

    Returns:
        处理结果
    """
    start_time = datetime.now()

    try:
        # 记录请求日志
        request_id = str(uuid.uuid4())
        default_logger.info(f"请求ID: {request_id}, 查询: {query}")

        # 调用Agent Core
        result = agent.run(query)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # 记录响应日志
        if result['success']:
            default_logger.info(
                f"响应成功, 耗时: {duration:.2f}s, "
                f"步骤数: {result.get('steps', 0)}"
            )
        else:
            default_logger.error(
                f"响应失败, 耗时: {duration:.2f}s, "
                f"错误: {result.get('error', 'unknown')}"
            )

        return result

    except Exception as e:
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        default_logger.error(f"处理异常, 耗时: {duration:.2f}s, 错误: {e}")
        return {"success": False, "error": str(e)}


def stream_process_query(
    query: str,
    agent: AgentCore
) -> Generator[str, None, None]:
    """
    流式处理查询

    Args:
        query: 用户查询
        agent: Agent Core实例

    Yields:
        生成Token字符
    """
    start_time = datetime.now()
    request_id = str(uuid.uuid4())

    try:
        default_logger.info(f"流式请求ID: {request_id}, 查询: {query}")

        yield "[STREAM_START]"

        full_answer = []
        final_answer = None

        try:
            for event in agent.stream_run(query):
                if event.get("event") == "error":
                    error_msg = event.get("error", "未知错误")
                    default_logger.error(f"流式执行错误: {error_msg}")
                    yield f"\n[ERROR] {error_msg}"
                    break

                for node_name, node_output in event.items():
                    if node_name == "__end__":
                        continue

                    if isinstance(node_output, dict):
                        if "final_answer" in node_output and node_output["final_answer"]:
                            if final_answer is None:
                                final_answer = node_output["final_answer"]
                                for char in final_answer:
                                    yield char
                                    full_answer.append(char)

                        if "error" in node_output and node_output["error"]:
                            error_msg = node_output["error"]
                            default_logger.error(f"节点 {node_name} 错误: {error_msg}")
                            yield f"\n[ERROR] {error_msg}"

                        if "step_count" in node_output:
                            default_logger.info(
                                f"流式执行节点: {node_name}, "
                                f"步骤: {node_output['step_count']}"
                            )

        except Exception as stream_error:
            default_logger.error(f"流式执行异常: {stream_error}")
            yield f"\n[ERROR] 流式执行失败: {str(stream_error)}"

        yield "\n[STREAM_END]"

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        default_logger.info(
            f"流式请求完成, 耗时: {duration:.2f}s, "
            f"总字符数: {len(full_answer)}"
        )

    except Exception as e:
        default_logger.error(f"流式处理失败: {e}")
        yield f"\n[ERROR] 处理失败: {str(e)}"


def handle_global_error(error: Exception) -> str:
    """
    全局错误处理

    Args:
        error: 异常对象

    Returns:
        友好错误提示
    """
    default_logger.error(f"全局错误: {type(error).__name__}: {error}", exc_info=True)

    error_type = type(error).__name__

    # 根据错误类型返回友好提示
    if "Connection" in error_type or "Timeout" in error_type:
        return "网络连接超时，请检查网络连接后重试。"
    elif "ValueError" in error_type:
        return f"参数错误: {str(error)}"
    elif "KeyError" in error_type:
        return "系统配置错误，请联系管理员。"
    elif "RuntimeError" in error_type:
        return "系统运行时错误，请联系管理员。"
    else:
        return f"系统错误: {error_type}，请稍后重试或联系管理员。"


def setup_signal_handlers() -> None:
    """
    设置信号处理器
    """
    try:
        import signal

        def signal_handler(signum, frame):
            """信号处理函数"""
            signal_name = {
                2: 'SIGINT',
                15: 'SIGTERM'
            }.get(signum, f'信号{signum}')
            default_logger.info(f"收到信号: {signal_name} ({signum})")
            print("\n程序被中断，正在清理资源...")
            cleanup()
            sys.exit(0)

        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            default_logger.info("信号处理器设置成功")
        except ValueError as e:
            if sys.platform == 'win32':
                default_logger.warning("Windows平台不支持SIGTERM信号处理")
                try:
                    signal.signal(signal.SIGINT, signal_handler)
                    default_logger.info("SIGINT信号处理器设置成功")
                except Exception as win_e:
                    default_logger.warning(f"Windows信号处理器设置失败: {win_e}")
            else:
                raise e

    except Exception as e:
        default_logger.warning(f"信号处理器设置失败: {e}")


def cleanup() -> None:
    """
    清理资源
    """
    default_logger.info("开始清理资源...")
    cleanup_success = True

    try:
        if HAS_CONFIG:
            from config import config
            default_logger.info("1. 清理Elasticsearch连接...")
            try:
                if hasattr(config, '_es_client') and config._es_client is not None:
                    config._es_client.close()
                    config._es_client = None
                    default_logger.info("   Elasticsearch连接已关闭")
            except Exception as e:
                default_logger.warning(f"   Elasticsearch清理失败: {e}")
                cleanup_success = False
        else:
            try:
                from rag.retriever import rag_retriever
                if hasattr(rag_retriever, 'es_client') and rag_retriever.es_client is not None:
                    rag_retriever.es_client.close()
                    rag_retriever.es_client = None
                    default_logger.info("   Elasticsearch连接已关闭")
            except Exception as e:
                default_logger.warning(f"   Elasticsearch清理失败: {e}")
                cleanup_success = False

    except ImportError:
        pass

    try:
        from mcp.scheduler import skill_scheduler
        default_logger.info("2. 清理技能调度器资源...")
        try:
            if hasattr(skill_scheduler, 'shutdown') and callable(skill_scheduler.shutdown):
                skill_scheduler.shutdown()
                default_logger.info("   技能调度器已关闭")
        except Exception as e:
            default_logger.warning(f"   技能调度器清理失败: {e}")
            cleanup_success = False
    except ImportError:
        pass

    try:
        from rag.retriever import rag_retriever
        default_logger.info("3. 清理向量模型资源...")
        try:
            if hasattr(rag_retriever, '_embedding_model') and rag_retriever._embedding_model is not None:
                del rag_retriever._embedding_model
                rag_retriever._embedding_model = None
                default_logger.info("   向量模型资源已释放")
        except Exception as e:
            default_logger.warning(f"   向量模型清理失败: {e}")
            cleanup_success = False

        try:
            if hasattr(rag_retriever, 'es_client') and rag_retriever.es_client is not None:
                rag_retriever.es_client.close()
                rag_retriever.es_client = None
                default_logger.info("   RAG检索器ES连接已关闭")
        except Exception as e:
            default_logger.warning(f"   RAG检索器ES清理失败: {e}")
            cleanup_success = False
    except ImportError:
        pass

    try:
        from agent.core import agent_core
        default_logger.info("4. 清理Agent Core资源...")
        try:
            if hasattr(agent_core, '_graph') and agent_core._graph is not None:
                agent_core._graph = None
            if hasattr(agent_core, '_llm') and agent_core._llm is not None:
                agent_core._llm = None
            agent_core._initialized = False
            default_logger.info("   Agent Core资源已清理")
        except Exception as e:
            default_logger.warning(f"   Agent Core清理失败: {e}")
            cleanup_success = False
    except ImportError:
        pass

    try:
        default_logger.info("5. 清理FastText意图识别模型...")
        try:
            from intent.recognizer import intent_recognizer
            if hasattr(intent_recognizer, '_model') and intent_recognizer._model is not None:
                del intent_recognizer._model
                intent_recognizer._model = None
                default_logger.info("   意图识别模型已释放")
        except Exception as e:
            default_logger.warning(f"   意图识别模型清理失败: {e}")
            cleanup_success = False
    except (ImportError, AttributeError):
        pass

    try:
        default_logger.info("6. 清理记忆系统...")
        try:
            from agent.memory import memory_system
            if hasattr(memory_system, '_conversation_memory') and memory_system._conversation_memory is not None:
                memory_system._conversation_memory.clear()
            if hasattr(memory_system, '_long_term_memory') and memory_system._long_term_memory is not None:
                memory_system._long_term_memory.clear()
            default_logger.info("   记忆系统已清理")
        except Exception as e:
            default_logger.warning(f"   记忆系统清理失败: {e}")
            cleanup_success = False
    except (ImportError, AttributeError):
        pass

    if cleanup_success:
        default_logger.info("资源清理完成")
    else:
        default_logger.warning("资源清理完成（部分失败）")


if __name__ == '__main__':
    # 设置信号处理器
    setup_signal_handlers()

    try:
        # 运行主函数
        main()
    except Exception as e:
        # 全局错误处理
        error_msg = handle_global_error(e)
        print(f"\n{error_msg}")
        import traceback
        traceback.print_exc()

    finally:
        # 清理资源
        cleanup()
