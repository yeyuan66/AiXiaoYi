"""
MCP协议与Skill调度器测试脚本
"""

import sys
import json
from typing import Optional

# 添加项目路径
sys.path.insert(0, '.')

from mcp.protocol import (
    MCPRequest,
    MCPResponse,
    MCPError,
    MCPProtocolValidator,
    RequestStatus
)
from mcp.skill_base import SkillBase
from mcp.scheduler import SkillScheduler, skill_scheduler
from mcp.skills.echo_skill import EchoSkill


def test_mcp_protocol() -> None:
    """测试MCP协议"""
    print("=" * 50)
    print("MCP协议测试")
    print("=" * 50)

    # 1. 测试请求创建
    print("\n1. 测试MCP请求创建...")
    request = MCPRequest(
        skill_name="echo",
        params={"message": "Hello MCP!"},
        trace_id="test_trace_001"
    )

    print(f"   Skill名称: {request.skill_name}")
    print(f"   参数: {request.params}")
    print(f"   Trace ID: {request.trace_id}")
    print(f"   请求ID: {request.request_id}")

    # 2. 测试请求验证
    print("\n2. 测试请求验证...")
    is_valid, error = request.validate()
    print(f"   验证结果: {'有效' if is_valid else '无效'}")
    if error:
        print(f"   错误信息: {error}")

    # 3. 测试请求序列化
    print("\n3. 测试请求序列化...")
    request_dict = request.to_dict()
    print("   字典格式:")
    dict_str = json.dumps(request_dict, ensure_ascii=False, indent=4)
    print("   " + dict_str[:200] + "...")

    request_json = request.to_json()
    print(f"\n   JSON长度: {len(request_json)}字符")

    # 4. 测试请求反序列化
    print("\n4. 测试请求反序列化...")
    request_restored = MCPRequest.from_json(request_json)
    print(f"   反序列化成功: {request_restored.skill_name == request.skill_name}")

    # 5. 测试响应创建（成功）
    print("\n5. 测试MCP响应创建（成功）...")
    response = MCPResponse.success_response(
        data={"result": "操作成功"},
        trace_id=request.trace_id,
        request_id=request.request_id,
        execution_time=123.45
    )

    print(f"   成功状态: {response.success}")
    print(f"   响应状态: {response.status}")
    print(f"   执行时间: {response.execution_time}ms")

    # 6. 测试响应创建（错误）
    print("\n6. 测试MCP响应创建（错误）...")
    error_response = MCPResponse.error_response(
        error="参数验证失败",
        trace_id=request.trace_id,
        request_id=request.request_id
    )

    print(f"   成功状态: {error_response.success}")
    print(f"   响应状态: {error_response.status}")
    print(f"   错误信息: {error_response.error}")

    # 7. 测试trace_id一致性验证
    print("\n7. 测试trace_id一致性验证...")
    is_consistent, error = MCPProtocolValidator.validate_trace_id_consistency(
        request,
        response
    )
    print(f"   Trace ID一致: {is_consistent}")
    if error:
        print(f"   错误信息: {error}")

    # 8. 测试错误定义
    print("\n8. 测试MCP错误定义...")
    error_obj = MCPError(
        code=MCPError.SKILL_NOT_FOUND,
        message="Skill不存在",
        detail="echo_skill未注册"
    )
    print(f"   错误代码: {error_obj.code}")
    print(f"   错误消息: {error_obj.message}")
    print(f"   错误详情: {error_obj.detail}")
    print(f"   字符串表示: {error_obj}")


def test_skill_base() -> None:
    """测试Skill基类"""
    print("\n" + "=" * 50)
    print("Skill基类测试")
    print("=" * 50)

    # 1. 测试Skill创建
    print("\n1. 测试Skill创建...")
    echo_skill = EchoSkill()
    print(f"   Skill名称: {echo_skill.name}")
    print(f"   Skill版本: {echo_skill.version}")
    print(f"   初始化状态: {echo_skill.health_check()}")

    # 2. 测试Skill信息
    print("\n2. 测试Skill信息...")
    info = echo_skill.get_info()
    print("   Skill信息:")
    for key, value in info.items():
        print(f"     {key}: {value}")

    # 3. 测试参数schema
    print("\n3. 测试参数schema...")
    schema = echo_skill.get_param_schema()
    print("   参数schema:")
    for param_name, param_info in schema.items():
        print(f"     {param_name}:")
        for key, value in param_info.items():
            print(f"       {key}: {value}")


def test_skill_scheduler() -> None:
    """测试Skill调度器"""
    print("\n" + "=" * 50)
    print("Skill调度器测试")
    print("=" * 50)

    # 1. 测试Skill注册
    print("\n1. 测试Skill注册...")
    echo_skill = EchoSkill()
    success = skill_scheduler.register_skill(echo_skill)
    print(f"   EchoSkill注册: {'成功' if success else '失败'}")

    # 2. 测试Skill查询
    print("\n2. 测试Skill查询...")
    skill = skill_scheduler.get_skill("echo")
    print(f"   获取echo Skill: {'成功' if skill is not None else '失败'}")

    # 3. 测试Skill列表
    print("\n3. 测试Skill列表...")
    skills = skill_scheduler.list_skills()
    print(f"   已注册Skill数量: {len(skills)}")
    for skill_name, skill_info in skills.items():
        print(f"     - {skill_name}: {skill_info['version']}")

    # 4. 测试Skill执行（同步）
    print("\n4. 测试Skill执行（同步）...")
    request = MCPRequest(
        skill_name="echo",
        params={"message": "Hello Scheduler!"}
    )

    response = skill_scheduler.execute(request)
    print(f"   执行结果: {'成功' if response.success else '失败'}")
    print(f"   Trace ID: {response.trace_id}")
    print(f"   执行时间: {response.execution_time:.2f}ms")
    if response.success:
        print(f"   响应数据: {response.data}")

    # 5. 测试Skill执行（带延迟）
    print("\n5. 测试Skill执行（带延迟）...")
    request_delay = MCPRequest(
        skill_name="echo",
        params={"message": "Delayed test", "delay": 500}
    )

    response_delay = skill_scheduler.execute(request_delay)
    print(f"   执行结果: {'成功' if response_delay.success else '失败'}")
    print(f"   执行时间: {response_delay.execution_time:.2f}ms")

    # 6. 测试Skill不存在的情况
    print("\n6. 测试Skill不存在的情况...")
    request_not_found = MCPRequest(
        skill_name="nonexistent_skill",
        params={}
    )

    response_not_found = skill_scheduler.execute(request_not_found)
    print(f"   执行结果: {'成功' if response_not_found.success else '失败'}")
    if not response_not_found.success:
        print(f"   错误信息: {response_not_found.error}")

    # 7. 测试批量执行
    print("\n7. 测试批量执行...")
    batch_requests = [
        MCPRequest(skill_name="echo", params={"message": f"Batch test {i}"})
        for i in range(3)
    ]

    batch_responses = skill_scheduler.batch_execute(batch_requests)
    print(f"   批量执行: {len(batch_responses)}个请求")
    for i, resp in enumerate(batch_responses, 1):
        print(f"     请求{i}: {'成功' if resp.success else '失败'}, "
              f"耗时: {resp.execution_time:.2f}ms")

    # 8. 测试异步执行
    print("\n8. 测试异步执行...")

    async_results = []
    def async_callback(response: MCPResponse) -> None:
        async_results.append(response)
        print(f"     异步回调: {response.data.get('message', 'N/A')}")

    request_async = MCPRequest(
        skill_name="echo",
        params={"message": "Async test"}
    )

    task_id = skill_scheduler.execute_async(request_async, async_callback)
    print(f"   异步任务ID: {task_id}")

    # 等待异步完成
    import time
    time.sleep(0.5)
    print(f"   异步结果: {len(async_results)}个")

    # 9. 测试健康检查
    print("\n9. 测试健康检查...")
    health = skill_scheduler.health_check()
    print("   健康状态:")
    for key, value in health.items():
        if key != "details":
            print(f"     {key}: {value}")

    # 10. 测试Skill自动发现
    print("\n10. 测试Skill自动发现...")
    discovered = skill_scheduler.discover_skills("mcp.skills")
    print(f"   自动发现Skill数量: {discovered}")

    # 11. 测试Skill注销
    print("\n11. 测试Skill注销...")
    unregistered = skill_scheduler.unregister_skill("echo")
    print(f"   注销echo Skill: {'成功' if unregistered else '失败'}")

    # 重新注册用于后续测试
    skill_scheduler.register_skill(echo_skill)


def test_concurrent_execution() -> None:
    """测试并发执行"""
    print("\n" + "=" * 50)
    print("并发执行测试")
    print("=" * 50)

    # 创建多个并发请求
    print("\n1. 创建并发请求...")
    concurrent_requests = [
        MCPRequest(
            skill_name="echo",
            params={"message": f"Concurrent test {i}", "delay": 200}
        )
        for i in range(5)
    ]

    print(f"   请求数量: {len(concurrent_requests)}")

    # 执行并发请求
    print("\n2. 执行并发请求...")
    import time
    start_time = time.time()

    concurrent_responses = skill_scheduler.batch_execute(concurrent_requests)

    total_time = time.time() - start_time

    print(f"   总耗时: {total_time:.2f}s")
    print(f"   平均耗时: {total_time/len(concurrent_requests):.2f}s")

    # 检查结果
    print("\n3. 检查结果...")
    all_success = all(r.success for r in concurrent_responses)
    print(f"   全部成功: {all_success}")

    for i, resp in enumerate(concurrent_responses, 1):
        print(f"     请求{i}: {resp.execution_time:.2f}ms, "
              f"Trace ID: {resp.trace_id}")


def main() -> None:
    """主测试函数"""
    print("\n")
    print("*" * 50)
    print("AI小益 MCP协议与Skill调度器测试")
    print("*" * 50)

    try:
        # 运行所有测试
        test_mcp_protocol()
        test_skill_base()
        test_skill_scheduler()
        test_concurrent_execution()

        print("\n" + "=" * 50)
        print("所有测试完成")
        print("=" * 50)

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
