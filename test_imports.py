"""
测试所有模块的导入
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    print("=" * 60)
    print("AI小益模块导入测试")
    print("=" * 60)

    # 测试列表
    modules = [
        ("config", "config.py"),
        ("utils", "utils"),
        ("utils.desensitize", "utils/desensitize.py"),
        ("utils.token_counter", "utils/token_counter.py"),
        ("utils.logger", "utils/logger.py"),

        ("intent", "intent.recognizer"),
        ("intent.recognizer", "intent/recognizer.py"),

        ("rag", "rag"),
        ("rag.retriever", "rag/retriever.py"),

        ("mcp", "mcp"),
        ("mcp.protocol", "mcp/protocol.py"),
        ("mcp.skill_base", "mcp/skill_base.py"),
        ("mcp.scheduler", "mcp/scheduler.py"),

        ("mcp.skills", "mcp/skills"),
        ("mcp.skills.rag_skill", "mcp/skills/rag_skill.py"),
        ("mcp.skills.echo_skill", "mcp/skills/echo_skill.py"),
        ("mcp.skills.sql_skill", "mcp/skills/sql_skill.py"),

        ("codeact", "codeact"),
        ("codeact.generator", "codeact/generator.py"),
        ("codeact.validator", "codeact/validator.py"),

        ("agent", "agent"),
        ("agent.core", "agent/core.py"),
        ("agent.memory", "agent/memory.py"),
        ("agent.prompt", "agent/prompt.py"),
    ]

    results = []

    for name, path in modules:
        try:
            __import__(name)
            module = sys.modules[name]
            results.append(f"  {name}: OK")
        except ImportError as e:
            results.append(f"  {name}: FAIL - {e}")
        except Exception as e:
            results.append(f"  {name}: FAIL - {type(e).__name__}: {e}")

    print("\n导入结果:")
    print("-" * 60)

    success_count = 0
    for result in results:
        print(result)
        if ": OK" in result:
            success_count += 1

    print("-" * 60)
    print(f"\n成功导入: {success_count}/{len(modules)}")

    # 测试关键功能
    print("\n" + "=" * 60)
    print("关键功能测试")
    print("=" * 60)

    try:
        from config import config
        print(f"  配置初始化: OK")
        print(f"  模型: {config.llm_model}")
    except Exception as e:
        print(f"  配置初始化: FAIL - {e}")

    try:
        from intent.recognizer import intent_recognizer
        from intent.recognizer import IntentType
        print(f"  意图识别器初始化: OK")
        print(f"  意图类型: {[it.value for it in IntentType]}")
    except Exception as e:
        print(f"  意图识别器初始化: FAIL - {e}")

    try:
        from mcp.scheduler import skill_scheduler
        print(f"  技能调度器初始化: OK")
        skills = skill_scheduler.list_skills()
        print(f"  已注册技能: {list(skills.keys())}")
    except Exception as e:
        print(f"  技能调度器初始化: FAIL - {e}")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    test_imports()
