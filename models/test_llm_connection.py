from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

def test_llm_connection() -> None:
    # 加载 .env 配置
    load_dotenv()

    # 初始化火山引擎 GLM-4.7
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.1,
        timeout=30  # 超时时间，防止卡住
    )

    print("正在测试火山引擎 GLM-4.7 连接...")
    try:
        # 发送一个简单的测试请求
        response = llm.invoke("你能帮我做什么？")
        print("✅ 连接成功！模型返回结果：")
        print(response.content)
    except Exception as e:
        print("❌ 连接失败，错误信息：")
        print(type(e).__name__, str(e))

if __name__ == '__main__':
    test_llm_connection()