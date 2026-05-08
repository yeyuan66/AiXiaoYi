"""
意图识别模块
基于scikit-learn (TF-IDF + LogisticRegression) 实现用户query的意图分类
无编译依赖，可直接运行
"""

import os
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import joblib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    from config import config
    from utils import default_logger
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    default_logger = logging.getLogger('intent_recognizer')


class IntentType(Enum):
    """意图类型枚举"""
    KNOWLEDGE_QUERY = "knowledge_query"    # 知识问答
    DATA_QUERY = "data_query"            # 数据查询
    TASK_OPERATION = "task_operation"    # 任务操作
    UNKNOWN = "unknown"                  # 未知意图


@dataclass
class IntentResult:
    """意图识别结果"""
    intent: str               # 意图标签
    confidence: float         # 置信度 (0.0 - 1.0)
    clarification: Optional[str] = None  # 澄清话术


class IntentRecognizer:
    """
    意图识别器单例类
    使用sklearn的TF-IDF + LogisticRegression对用户query进行意图分类
    支持自动冷启动训练，无需外部模型文件
    """

    _instance: Optional['IntentRecognizer'] = None

    # 置信度阈值
    CONFIDENCE_THRESHOLD: float = 0.8

    # 模型保存路径
    MODEL_SAVE_PATH: str = "./models/intent_classifier.joblib"

    # 内置训练数据（冷启动用）
    TRAINING_DATA: dict[str, list[str]] = {
        "knowledge_query": [
            # 产品知识类
            "什么是固收类理财产品",
            "固收类产品有什么特点",
            "产品A的风险等级是什么",
            "如何选择理财产品",
            "理财产品的基本概念",
            "什么是权益类基金",
            "混合型基金的特点是什么",
            "产品的收益规则是怎样的",
            "风险等级的含义",
            "理财产品的投资范围",
            "什么是净值型产品",
            "如何理解产品的期限",
            "产品的起购金额要求",
            "理财产品有哪些类型",
            "什么是封闭式产品",
            "开放式和封闭式的区别",
            "产品的流动性如何",
            "如何评估产品风险",
            "产品说明书的解读",
            "理财产品的合规要求",
            # 业务知识类
            "什么叫收益升级",
            "中台系统是做什么的",
            "如何使用数据查询功能",
            "系统有哪些模块",
            "权限管理规则",
            "数据安全相关规定",
            "业务流程说明",
            "操作指南在哪里",
            "系统架构介绍",
            "数据字典说明",
            "接口文档在哪里",
            "如何配置环境",
        ],
        "data_query": [
            # 数据查询类
            "查询近30天产品收益",
            "统计上周销售总额",
            "查看本月产品销量",
            "查询用户购买记录",
            "统计收益率超过5%的产品",
            "查询昨天的大额交易",
            "统计各产品的购买人数",
            "查询产品A的净值数据",
            "统计季度收益排名",
            "查询用户余额信息",
            "统计不同风险等级的产品数量",
            "查询产品B的历史收益",
            "统计本月新增用户",
            "查询逾期产品列表",
            "统计产品赎回金额",
            "查询客户资产分布",
            "统计各渠道销售额",
            "查询产品持仓明细",
            "统计用户复购率",
            "查询最近一周的交易记录",
            # 数据分析类
            "分析产品收益趋势",
            "统计用户购买偏好",
            "查询销售数据报表",
            "分析客户画像",
            "统计产品表现数据",
            "查询业绩归因分析",
            "统计客户流失率",
            "查询转化率数据",
            "分析资金流向",
            "统计产品份额变动",
        ],
        "task_operation": [
            # 任务操作类
            "导出用户购买数据",
            "执行批量导入任务",
            "生成季度销售报表",
            "创建新的数据查询任务",
            "执行产品收益计算",
            "启动数据同步任务",
            "创建新的用户分组",
            "执行产品上架操作",
            "生成客户分析报告",
            "执行数据备份任务",
            "创建自动化任务",
            "执行审批流程",
            "生成月度业绩报告",
            "执行权限分配",
            "创建定时任务",
            "执行数据清洗",
            "生成对账单",
            "执行系统配置更新",
            "创建数据看板",
            "执行批量通知发送",
            # 数据处理类
            "处理用户申请",
            "执行数据校验",
            "生成统计图表",
            "执行数据迁移",
            "创建自定义报表",
            "执行批量修改",
            "生成预警通知",
            "执行数据归档",
            "创建数据快照",
            "执行数据恢复",
        ],
    }

    # 澄清话术模板
    CLARIFICATION_TEMPLATES: dict[str, str] = {
        "knowledge_query": "您是在咨询理财产品相关知识吗？我可以帮您查询产品说明、风险等级、收益规则等信息。",
        "data_query": "您是想查询某个产品的具体数据吗？请告诉我产品名称和时间范围。",
        "task_operation": "您是要执行某项操作任务吗？请说明具体需要做什么。",
        "unknown": "抱歉，我不太理解您的需求。您可以咨询理财产品知识、查询产品数据，或执行相关操作。",
    }

    def __new__(cls) -> 'IntentRecognizer':
        """实现单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """初始化意图识别器"""
        if not hasattr(self, '_initialized'):
            self.model: Optional[Pipeline] = None
            self._model_loaded = False
            self._intent_labels: list[str] = []

            self._load_or_train_model()
            self._initialized = True

    def _load_or_train_model(self) -> None:
        """加载模型或进行冷启动训练"""
        try:
            # 尝试加载已保存的模型
            if os.path.exists(self.MODEL_SAVE_PATH):
                self._load_model()
                if self._model_loaded:
                    default_logger.info(
                        f"意图模型加载成功: {self.MODEL_SAVE_PATH}"
                    )
                    return

            # 模型不存在或加载失败，进行冷启动训练
            default_logger.info("开始冷启动训练意图模型...")
            self._train_model()

        except Exception as e:
            default_logger.error(f"模型初始化失败: {e}")
            self._model_loaded = False

    def _train_model(self) -> None:
        """使用内置数据进行冷启动训练"""
        try:
            # 准备训练数据
            X_train: list[str] = []
            y_train: list[str] = []

            for intent, queries in self.TRAINING_DATA.items():
                X_train.extend(queries)
                y_train.extend([intent] * len(queries))

            if len(X_train) == 0:
                raise ValueError("训练数据为空")

            # 创建训练数据增强（简单的同义词替换）
            X_train_enhanced, y_train_enhanced = self._augment_training_data(
                X_train, y_train
            )

            # 创建TF-IDF + LogisticRegression管道
            # 兼容不同版本的sklearn
            self.model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    analyzer='char_wb',      # 字符级n-gram，对中文更友好
                    ngram_range=(2, 4),     # 2-4元组
                    max_features=2000,      # 限制特征数量
                    min_df=1,
                    max_df=0.9,
                    lowercase=True
                )),
                ('clf', LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    solver='lbfgs',
                    random_state=42
                ))
            ])

            # 训练模型
            self.model.fit(X_train_enhanced, y_train_enhanced)
            self._intent_labels = self.model.classes_.tolist()
            self._model_loaded = True

            # 保存模型
            self._save_model()

            default_logger.info(
                f"意图模型训练完成，类别: {self._intent_labels}，"
                f"训练样本数: {len(X_train_enhanced)}"
            )

        except Exception as e:
            default_logger.error(f"模型训练失败: {e}")
            self._model_loaded = False

    def _augment_training_data(
        self,
        X: list[str],
        y: list[str]
    ) -> tuple[list[str], list[str]]:
        """
        简单的训练数据增强

        Args:
            X: 原始文本
            y: 原始标签

        Returns:
            tuple: 增强后的文本和标签
        """
        X_aug = X.copy()
        y_aug = y.copy()

        # 简单的同义词替换映射
        synonyms = {
            "查询": ["查", "查看", "检索", "获取"],
            "统计": ["计算", "分析", "汇总", "统计"],
            "产品": ["理财", "基金", "标的"],
            "收益": ["回报", "利润", "收益"],
            "用户": ["客户", "投资人", "投资者"],
            "导出": ["下载", "保存", "导出"],
            "执行": ["运行", "启动", "执行"],
            "生成": ["创建", "制作", "生成"],
            "近": ["最近", "过去", "近"],
        }

        for text, label in zip(X, y):
            for original, replacements in synonyms.items():
                if original in text:
                    for replacement in replacements:
                        if replacement != original:
                            augmented_text = text.replace(original, replacement)
                            X_aug.append(augmented_text)
                            y_aug.append(label)

        return X_aug, y_aug

    def _load_model(self) -> None:
        """加载已保存的模型"""
        try:
            model_data = joblib.load(self.MODEL_SAVE_PATH)
            self.model = model_data['model']
            self._intent_labels = model_data['labels']
            self._model_loaded = True
        except Exception as e:
            default_logger.error(f"模型加载失败: {e}")
            self._model_loaded = False

    def _save_model(self) -> None:
        """保存模型到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.MODEL_SAVE_PATH), exist_ok=True)

            model_data = {
                'model': self.model,
                'labels': self._intent_labels
            }
            joblib.dump(model_data, self.MODEL_SAVE_PATH)
            default_logger.info(f"模型已保存: {self.MODEL_SAVE_PATH}")
        except Exception as e:
            default_logger.warning(f"模型保存失败: {e}")

    def predict(
        self,
        query: str,
        threshold: Optional[float] = None
    ) -> IntentResult:
        """
        预测用户query的意图

        Args:
            query: 用户输入的自然语言query
            threshold: 置信度阈值，默认使用类默认值0.8

        Returns:
            IntentResult: 包含意图标签、置信度和澄清话术
        """
        if not query or not query.strip():
            return IntentResult(
                intent=IntentType.UNKNOWN.value,
                confidence=0.0,
                clarification=self.CLARIFICATION_TEMPLATES["unknown"]
            )

        threshold = threshold if threshold is not None else self.CONFIDENCE_THRESHOLD

        if self._model_loaded and self.model is not None:
            return self._predict_with_model(query, threshold)
        else:
            return self._predict_with_rules(query, threshold)

    def _predict_with_model(
        self,
        query: str,
        threshold: float
    ) -> IntentResult:
        """
        使用sklearn模型预测意图

        Args:
            query: 用户输入
            threshold: 置信度阈值

        Returns:
            IntentResult: 意图识别结果
        """
        try:
            # 获取预测结果
            intent = self.model.predict([query])[0]

            # 获取预测概率
            proba = self.model.predict_proba([query])[0]
            confidence = float(proba.max())

            # 判断是否需要澄清
            clarification = None
            if intent == IntentType.UNKNOWN.value or confidence < threshold:
                # 根据置信度返回对应的澄清话术
                if confidence >= 0.6:
                    clarification = self.CLARIFICATION_TEMPLATES.get(intent)
                else:
                    clarification = self.CLARIFICATION_TEMPLATES["unknown"]

            return IntentResult(
                intent=intent,
                confidence=confidence,
                clarification=clarification
            )

        except Exception as e:
            default_logger.error(f"模型预测失败: {e}")
            return self._predict_with_rules(query, threshold)

    def _predict_with_rules(
        self,
        query: str,
        threshold: float
    ) -> IntentResult:
        """
        规则回退方案：使用关键词规则进行意图识别

        Args:
            query: 用户输入
            threshold: 置信度阈值

        Returns:
            IntentResult: 意图识别结果
        """
        query_lower = query.lower()

        # 知识问答关键词
        knowledge_keywords = [
            '什么是', '什么叫', '如何', '怎么', '什么意思',
            '说明', '介绍', '规则', '原理', '特点',
            '风险', '收益', '定义', '概念', '区别',
            '解读', '理解', '含义', '要求', '范围'
        ]

        # 数据查询关键词
        data_keywords = [
            '查询', '查', '统计', '多少', '总额', '总数',
            '收益', '净值', '日期', '时间', '范围',
            '近', '最近', '过去', '到', '截至',
            '数据', '报表', '记录', '明细', '列表'
        ]

        # 任务任务关键词
        task_keywords = [
            '执行', '运行', '启动', '创建', '删除', '修改',
            '更新', '导出', '生成', '计算', '分析',
            '任务', '操作', '处理', '流程', '审批'
        ]

        # 计算各意图的匹配分数
        knowledge_score = sum(
            1 for kw in knowledge_keywords if kw in query_lower
        )
        data_score = sum(1 for kw in data_keywords if kw in query_lower)
        task_score = sum(1 for kw in task_keywords if kw in query_lower)

        # 确定意图
        scores = {
            IntentType.KNOWLEDGE_QUERY.value: knowledge_score,
            IntentType.DATA_QUERY.value: data_score,
            IntentType.TASK_OPERATION.value: task_score,
        }

        max_score = max(scores.values())
        max_intent = max(scores, key=scores.get)

        if max_score == 0:
            intent = IntentType.UNKNOWN.value
            confidence = 0.0
        else:
            # 归一化置信度 (最高到1.0)
            confidence = min(max_score / 3.0 * 0.9, 1.0)
            intent = max_intent

        # 判断是否需要澄清
        clarification = None
        if intent == IntentType.UNKNOWN.value or confidence < threshold:
            if confidence >= 0.6:
                clarification = self.CLARIFICATION_TEMPLATES.get(intent)
            else:
                clarification = self.CLARIFICATION_TEMPLATES["unknown"]

        return IntentResult(
            intent=intent,
            confidence=confidence,
            clarification=clarification
        )

    def batch_predict(
        self,
        queries: list[str],
        threshold: Optional[float] = None
    ) -> list[IntentResult]:
        """
        批量预测意图

        Args:
            queries: 用户query列表
            threshold: 置信度阈值

        Returns:
            list[IntentResult]: 意图识别结果列表
        """
        return [self.predict(q, threshold) for q in queries]

    def get_model_info(self) -> dict[str, any]:
        """
        获取模型信息

        Returns:
            dict: 模型信息字典
        """
        return {
            'loaded': self._model_loaded,
            'intent_labels': self._intent_labels,
            'model_path': self.MODEL_SAVE_PATH,
            'training_samples': sum(
                len(queries) for queries in self.TRAINING_DATA.values()
            ),
            'threshold': self.CONFIDENCE_THRESHOLD
        }

    def force_retrain(self) -> bool:
        """
        强制重新训练模型

        Returns:
            bool: 是否训练成功
        """
        default_logger.info("开始强制重新训练模型...")
        self._train_model()
        return self._model_loaded


# 创建全局意图识别器实例
intent_recognizer = IntentRecognizer()


# ============= 测试用例 =============
def test_intent_recognizer() -> None:
    """
    测试意图识别器
    """
    print("=" * 50)
    print("AI小益意图识别模块测试 (sklearn版本)")
    print("=" * 50)

    # 显示模型信息
    recognizer = IntentRecognizer()
    model_info = recognizer.get_model_info()
    print("\n模型信息:")
    print(f"  模型状态: {'已加载' if model_info['loaded'] else '未加载'}")
    print(f"  意图标签: {model_info['intent_labels']}")
    print(f"  训练样本数: {model_info['training_samples']}")
    print(f"  置信度阈值: {model_info['threshold']}")

    # 测试用例
    test_queries = [
        "什么是固收类理财产品？",
        "近30天固收类产品总收益是多少？",
        "导出上周的用户购买记录",
        "帮我统计本月的销售额",
        "产品A的风险等级是什么？",
        "查询产品B的净值走势",
        "创建一个新的销售报表任务",
        "混合型基金有什么特点？",
        "不知道我要做什么",
        "股票型基金的收益规则",
        "查一下昨天的交易数据",
        "生成客户分析报告",
        "如何评估产品风险",
        "统计用户复购率",
        "执行数据备份任务",
    ]

    print("\n测试用例:")
    print("-" * 50)

    for i, query in enumerate(test_queries, 1):
        result = recognizer.predict(query)

        print(f"\n测试 {i}: {query}")
        print(f"  意图: {result.intent}")
        print(f"  置信度: {result.confidence:.3f}")
        if result.clarification:
            print(f"  澄清: {result.clarification}")
        print("-" * 50)

    # 测试边界情况
    print("\n边界情况测试:")
    print("-" * 50)

    edge_cases = [
        ("", "空query"),
        ("   ", "纯空格"),
        ("123456", "纯数字"),
        ("!@#$%", "纯符号"),
        ("a" * 500, "超长文本"),
    ]

    for query, desc in edge_cases:
        result = recognizer.predict(query)
        print(f"\n{desc}: '{query[:30]}{'...' if len(query) > 30 else ''}'")
        print(f"  意图: {result.intent}")
        print(f"  置信度: {result.confidence:.3f}")
        print("-" * 50)

    # 测试批量预测
    print("\n批量预测测试:")
    print("-" * 50)

    batch_queries = [
        "什么是权益类产品？",
        "查询产品收益",
        "执行批量导出",
    ]

    batch_results = recognizer.batch_predict(batch_queries)

    for query, result in zip(batch_queries, batch_results):
        print(f"\nQuery: {query}")
        print(f"  意图: {result.intent}")
        print(f"  置信度: {result.confidence:.3f}")
        print("-" * 50)

    # 测试置信度阈值
    print("\n置信度阈值测试:")
    print("-" * 50)

    test_query = "什么是基金"
    thresholds = [0.5, 0.7, 0.8, 0.9]

    for threshold in thresholds:
        result = recognizer.predict(test_query, threshold=threshold)
        print(f"\n阈值 {threshold}:")
        print(f"  意图: {result.intent}")
        print(f"  置信度: {result.confidence:.3f}")
        print(f"  是否澄清: {result.clarification is not None}")
        print("-" * 50)


if __name__ == '__main__':
    test_intent_recognizer()
