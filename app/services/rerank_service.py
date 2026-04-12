"""重排序服务模块 - 基于阿里云 DashScope qwen3-rerank 模型"""

from typing import List

import dashscope
from dashscope import TextReRank
from loguru import logger

from app.config import config


class RerankService:
    """
    重排序服务 (Reranker)

    职责：接收【查询问题】和【候选文档列表】，调用 qwen3-rerank 模型，
    对每个候选文档与查询问题之间的相关性打分，并返回附带分数信息的结果列表。

    与向量检索的核心区别：
    - 向量检索（双塔）：问题和文档分别独立编码，仅比较几何距离。
    - Rerank（交叉编码）：将问题和文档拼接后共同输入模型，利用完整注意力
      机制深度分析语义相关性，精度更高但速度更慢。
    """

    def __init__(self, api_key: str, model: str = "qwen3-rerank") -> None:
        """
        初始化重排序服务

        Args:
            api_key: DashScope API Key
            model: 重排序模型名称，默认使用 qwen3-rerank
        """
        if not api_key or api_key == "your-api-key-here":
            raise ValueError("请设置环境变量 DASHSCOPE_API_KEY 以启用重排序服务")

        # DashScope TextReRank 使用原生 SDK，直接设置全局 api_key
        dashscope.api_key = api_key
        self.model = model

        # 打印初始化信息（对 Key 做脱敏处理）
        masked_key = self._mask_api_key(api_key)
        logger.info(
            f"RerankService 初始化完成 - 模型: {model}, API Key: {masked_key}"
        )

    @staticmethod
    def _mask_api_key(api_key: str) -> str:
        """对 API Key 进行脱敏，仅保留首尾几位，用于日志输出"""
        if len(api_key) > 8:
            return f"{api_key[:8]}...{api_key[-4:]}"
        return "***"

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int | None = None,
    ) -> List[dict]:
        """
        对候选文档列表按与查询问题的相关性进行重新打分和排序

        Args:
            query: 用户原始查询文本
            documents: 候选文档的原始文本列表（与 Milvus 召回结果一一对应）
            top_n: 可选。模型侧最多返回 top_n 个结果；为 None 时返回全部文档的分数。
                   注意：调用方应再做阈值过滤和数量截断，这里不做业务层限制。

        Returns:
            List[dict]: 按相关性分数降序排列的结果列表，每个元素包含：
                - index (int): 对应原始 documents 列表中的索引
                - relevance_score (float): 相关性分数，范围 [0, 1]，越大越相关

        Raises:
            RuntimeError: API 调用失败时抛出，附带详细错误信息
        """
        if not query or not query.strip():
            raise ValueError("Rerank 的查询文本不能为空")
        if not documents:
            logger.warning("Rerank 收到空文档列表，直接返回空结果")
            return []

        try:
            logger.info(
                f"开始重排序 - 查询: '{query[:50]}...', "
                f"候选文档数: {len(documents)}, top_n: {top_n}"
            )

            # 调用 DashScope TextReRank 原生接口
            response = TextReRank.call(
                model=self.model,
                query=query,
                documents=documents,
                top_n=top_n,          # None 时 API 会返回所有文档的分数
                return_documents=False,  # 在调用方已持有原始文档，无需重复返回
            )

            # 检查 API 响应状态
            if response.status_code != 200:
                error_msg = (
                    f"Rerank API 调用失败 - "
                    f"HTTP {response.status_code}, "
                    f"Code: {response.code}, "
                    f"Message: {response.message}"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            # 解析结果
            results = [
                {
                    "index": item.index,
                    "relevance_score": item.relevance_score,
                }
                for item in response.output.results
            ]

            # 按分数降序排列（通义模型通常已排好序，这里做防御性排序）
            results.sort(key=lambda x: x["relevance_score"], reverse=True)

            logger.info(
                f"重排序完成 - 返回 {len(results)} 个结果, "
                f"最高分: {results[0]['relevance_score']:.4f}, "
                f"最低分: {results[-1]['relevance_score']:.4f}"
                if results else "重排序完成 - 无结果"
            )

            return results

        except RuntimeError:
            # 直接向上传播已经格式化好的运行时异常
            raise
        except Exception as e:
            error_msg = f"Rerank 服务调用异常: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e


# 全局单例 - 与项目中其他 service 保持一致的单例模式
rerank_service = RerankService(
    api_key=config.dashscope_api_key,
    model=config.rerank_model,
)
