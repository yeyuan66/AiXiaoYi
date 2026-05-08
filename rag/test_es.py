# es_test.py
from elasticsearch import Elasticsearch


def test_es_connection():
    """
    测试 Elasticsearch 7.17.0 本地连接
    确保 ES 已启动在 http://127.0.0.1:9200
    """
    try:
        # 初始化 ES 客户端（本地无密码模式）
        es = Elasticsearch(
            hosts="http://127.0.0.1:9200",
            request_timeout=10
        )

        # 测试连接
        is_connected = es.ping()

        if is_connected:
            print("✅ Elasticsearch 连接成功！")
            print(f"ES 版本信息：{es.info()['version']['number']}")
        else:
            print("❌ Elasticsearch 连接失败！")

        return is_connected

    except Exception as e:
        print("❌ 连接异常：", str(e))
        return False


# 主函数入口
if __name__ == "__main__":
    test_es_connection()