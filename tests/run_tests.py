#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 小益 Agent Core 测试运行脚本

提供多种测试运行方式：
1. 运行所有测试
2. 运行指定模块测试
3. 生成测试报告
4. 查看测试覆盖率
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_pytest(args):
    """运行 pytest"""
    cmd = ["pytest", "-v"]

    # 添加参数
    if args.module:
        cmd.append(f"tests/test_{args.module}.py")

    if args.coverage:
        cmd.extend([
            "--cov=agent",
            "--cov-report=html",
            "--cov-report=term"
        ])

    if args.html_report:
        cmd.append("--html=reports/test_report.html")
        cmd.append("--self-contained-html")

    if args.marks:
        for mark in args.marks:
            cmd.append(f"-m {mark}")

    # 确保报告目录存在
    if args.coverage or args.html_report:
        Path("reports").mkdir(exist_ok=True)

    print("=" * 60)
    print("运行测试命令:")
    print(" ".join(cmd))
    print("=" * 60)

    result = subprocess.run(cmd)
    return result.returncode


def run_specific_test(test_file, test_name=None):
    """运行特定测试"""
    cmd = ["pytest", "-v", test_file]

    if test_name:
        cmd.append(f"-k {test_name}")

    print(f"运行测试: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def show_help():
    """显示帮助信息"""
    print("""
AI 小益 Agent Core 测试运行脚本

使用方法:
    python run_tests.py [选项]

选项:
    --all                    运行所有测试（默认）
    --module MODULE           运行指定模块测试 (core, memory, prompt)
    --test TEST_NAME         运行指定名称的测试
    --coverage               生成覆盖率报告
    --html-report            生成 HTML 测试报告
    --marks MARKS            按标记运行测试 (如: -m "unit")
    --list                   列出所有测试
    --help                   显示此帮助信息

示例:
    # 运行所有测试
    python run_tests.py

    # 运行 agent.core 模块测试
    python run_tests.py --module core

    # 生成覆盖率报告
    python run_tests.py --coverage

    # 只运行单元测试
    python run_tests.py --marks "unit"

    # 列出所有测试
    python run_tests.py --list
    """)


def list_tests():
    """列出所有测试"""
    print("\n可用的测试模块:")
    print("  - core:   Agent Core 核心功能测试")
    print("  - memory: 三级记忆系统测试")
    print("  - prompt: 提示词模板测试")
    print("\n运行所有测试:")
    print("  python run_tests.py --all")


def main():
    parser = argparse.ArgumentParser(
        description="AI 小益 Agent Core 测试运行工具"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="运行所有测试"
    )

    parser.add_argument(
        "--module",
        type=str,
        choices=["core", "memory", "prompt"],
        help="运行指定模块测试"
    )

    parser.add_argument(
        "--test",
        type=str,
        help="运行指定名称的测试"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="生成覆盖率报告"
    )

    parser.add_argument(
        "--html-report",
        action="store_true",
        help="生成 HTML 测试报告"
    )

    parser.add_argument(
        "--marks",
        nargs="*",
        help="按标记运行测试"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有测试"
    )

    args = parser.parse_args()

    # 处理列出测试
    if args.list:
        list_tests()
        return 0

    # 处理运行特定测试
    if args.test:
        return run_specific_test(f"tests/test_{args.test}.py")

    # 默认运行所有测试
    if not any([args.all, args.module]):
        args.all = True

    return run_pytest(args)


if __name__ == "__main__":
    sys.exit(main())
