"""向量检索服务模块 - 三路检索：向量粗排 + ES BM25 + RRF融合 + Rerank精排

检索链路：
    1. 【向量召回】 Milvus ANN 搜索，基于语义相似度召回 rough_top_k 个候选
    2. 【BM25召回】  Elasticsearch BM25 全文检索，基于关键词精确匹配召回 es_top_k 个候选
    3. 【RRF融合】   将两路结果按排名合并，计算 RRF 融合分数，最多保留 rrf_top 个
    4. 【Rerank精排】 qwen3-rerank 对融合结果做 Cross-Encoder 深度打分，截取最终 top_k 个
"""

from typing import Any, Dict, List

from loguru import logger
from pymilvus import Collection

from app.config import config
from app.core.milvus_client import milvus_manager
from app.services.rerank_service import rerank_service
from app.services.vector_embedding_service import vector_embedding_service
from app.services.es_search_service import es_search_service


class SearchResult:
    """搜索结果数据类"""

    def __init__(
        self,
        id: str,
        content: str,
        score: float,
        metadata: Dict[str, Any],
        rerank_score: float | None = None,
    ) -> None:
        """
        初始化搜索结果

        Args:
            id:           文档在 Milvus/ES 中共享的唯一 UUID
            content:      文档原始文本内容
            score:        原始检索分数（向量L2距离 或 RRF融合分数）
            metadata:     文档元数据（来源文件路径、标题等）
            rerank_score: Rerank 模型给出的相关性分数 [0, 1]
        """
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata
        self.rerank_score = rerank_score

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，供 API 返回或日志使用"""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "metadata": self.metadata,
        }


class VectorSearchService:
    """
    混合检索服务 - 四阶段检索策略

    检索流程：
        1. 【向量粗排】 Milvus ANN：基于语义相似度快速召回 rough_top_k 个候选。
        2. 【BM25召回】 ES 全文检索：基于关键词精确匹配召回 es_top_k 个候选。
        3. 【RRF融合】  对两路结果按排名加权融合，利用 doc_id 去重，
                        得到综合排名的候选集（≤ rrf_top 个）。
        4. 【Rerank精排】qwen3-rerank Cross-Encoder 对候选集精排，
                        最终截取 top_k 个高质量文档给 LLM。

    RRF 公式：
        score(doc) = Σ  1 / (60 + rank_i)
        - rank_i：doc 在第 i 路结果中的排名（从 1 开始）
        - 60：经验平滑常数，防止头部排名权重过大
        - 未出现在某路结果中的文档，该路贡献为 0
    """

    def __init__(self) -> None:
        """初始化混合检索服务"""
        logger.info("VectorSearchService 初始化完成（向量 + ES BM25 双路召回模式）")

    def search_similar_documents(
        self,
        query: str,
        top_k: int = config.rag_top_k,
        rough_top_k: int = config.rag_rough_top_k,
        es_top_k: int = config.es_top_k,
    ) -> List[SearchResult]:
        """
        四阶段混合检索：向量召回 + BM25召回 + RRF融合 + Rerank精排

        Args:
            query:        用户查询文本
            top_k:        最终返回给 LLM 的文档数量上限
            rough_top_k:  向量粗排召回的候选数量
            es_top_k:     ES BM25 召回的候选数量

        Returns:
            List[SearchResult]: 经过重排序后的结果列表，按 rerank_score 降序，最多 top_k 条
        """
        try:
            logger.info(
                f"开始混合检索 - 查询: '{query[:60]}...', "
                f"向量召回: {rough_top_k}, ES召回: {es_top_k}, Top-K: {top_k}"
            )

            # ──────────────────────────────────────────────
            # 阶段 1：向量粗排（Milvus ANN）
            # ──────────────────────────────────────────────
            vector_results = self._vector_search(query, rough_top_k)
            logger.info(f"[阶段1] 向量召回完成: {len(vector_results)} 个候选文档")

            # ──────────────────────────────────────────────
            # 阶段 2：ES BM25 关键词召回
            # ──────────────────────────────────────────────
            es_results = self._es_bm25_search(query, es_top_k)
            logger.info(f"[阶段2] ES BM25 召回完成: {len(es_results)} 个候选文档")

            # 如果两路均为空，直接返回
            if not vector_results and not es_results:
                logger.warning("两路召回均未命中任何文档，知识库可能为空")
                return []

            # ──────────────────────────────────────────────
            # 阶段3：RRF 分数融合（去重 + 合并排名）
            # ──────────────────────────────────────────────
            fused_results = self._rrf_fusion(
                vector_results,
                es_results,
                rrf_top=config.rrf_top,  # 融合后送入 Rerank 的最大候选数，来自配置文件
                k=config.rrf_k,          # RRF 平滑常数，来自配置文件
            )
            logger.info(f"[阶段3] RRF 融合完成: {len(fused_results)} 个候选文档")

            # ──────────────────────────────────────────────
            # 阶段 4：Rerank 精排（Cross-Encoder 深度打分）
            # ──────────────────────────────────────────────
            final_results = self._apply_rerank(
                query=query,
                rough_results=fused_results,
                top_k=top_k,
            )

            logger.info(
                f"混合检索完成 - 最终返回 {len(final_results)} 个文档"
            )
            return final_results

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            raise RuntimeError(f"检索失败: {e}") from e

    def _vector_search(self, query: str, rough_top_k: int) -> List[SearchResult]:
        """
        执行向量近似最近邻搜索（阶段1：向量粗排）

        Args:
            query:       查询文本
            rough_top_k: 需要召回的候选文档数量

        Returns:
            List[SearchResult]: 粗排结果，按 L2 距离升序（越小越相似），rerank_score 为 None
        """
        # 1. 将查询文本向量化
        query_vector = vector_embedding_service.embed_query(query)
        logger.debug(f"查询向量生成成功，维度: {len(query_vector)}")

        # 2. 获取 Milvus collection 实例
        collection: Collection = milvus_manager.get_collection()

        # 3. 构建搜索参数（IVF_FLAT + L2 距离）
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        # 4. 执行 ANN 搜索
        results = collection.search(
            data=[query_vector],
            anns_field="vector",
            param=search_params,
            limit=rough_top_k,
            output_fields=["id", "content", "metadata"],
        )

        # 5. 将 Milvus 原生结果解析为 SearchResult 对象
        rough_results: List[SearchResult] = []
        for hits in results:
            for hit in hits:
                rough_results.append(
                    SearchResult(
                        id=hit.entity.get("id"),
                        content=hit.entity.get("content"),
                        score=hit.distance,
                        metadata=hit.entity.get("metadata", {}),
                        rerank_score=None,
                    )
                )

        return rough_results

    def _es_bm25_search(self, query: str, es_top_k: int) -> List[SearchResult]:
        """
        执行 ES BM25 全文检索（阶段2：关键词召回）

        Args:
            query:    查询文本
            es_top_k: 需要召回的候选文档数量

        Returns:
            List[SearchResult]: BM25 召回结果，score 为 ES _score，rerank_score 为 None
        """
        raw_results = es_search_service.search(query, es_top_k)

        es_results: List[SearchResult] = []
        for item in raw_results:
            es_results.append(
                SearchResult(
                    id=item["doc_id"],
                    content=item["content"],
                    score=item["bm25_score"],   # ES BM25 原始分数
                    metadata=item["metadata"],
                    rerank_score=None,
                )
            )

        return es_results

    def _rrf_fusion(
        self,
        vector_results: List[SearchResult],
        es_results: List[SearchResult],
        rrf_top: int = config.rrf_top,  # 默认从配置文件读取
        k: int = config.rrf_k,          # 默认从配置文件读取
    ) -> List[SearchResult]:
        """
        RRF（Reciprocal Rank Fusion）分数融合

        对两路召回结果按排名加权合并，公式：
            score(doc) = Σ  1 / (k + rank_i)

        Args:
            vector_results: 向量召回结果（按 L2 距离升序）
            es_results:     ES BM25 召回结果（按 BM25 分数降序）
            rrf_top:        融合后保留的最大文档数量
            k:              RRF 平滑常数，默认 60（工业经验值）

        Returns:
            List[SearchResult]: RRF 融合后的结果，score 字段更新为 RRF 分数，降序排列
        """
        # rrf_scores: doc_id -> RRF 累积分数
        rrf_scores: Dict[str, float] = {}
        # doc_store: doc_id -> SearchResult 对象（用于后续返回）
        doc_store: Dict[str, SearchResult] = {}

        # 计算向量召回路的 RRF 贡献
        # 注意：Milvus L2 距离越小越相似，已经是升序，rank 与相关性正确对应
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_store:
                doc_store[doc_id] = result

        # 计算 ES BM25 路的 RRF 贡献
        # ES 返回的结果已按 _score 降序，rank 与相关性正确对应
        for rank, result in enumerate(es_results, start=1):
            doc_id = result.id
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_store:
                doc_store[doc_id] = result

        # 按 RRF 分数降序排列，截取前 rrf_top 个
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        sorted_ids = sorted_ids[:rrf_top]

        # 构造融合后的结果列表，将 score 更新为 RRF 分数
        fused_results: List[SearchResult] = []
        for doc_id in sorted_ids:
            result = doc_store[doc_id]
            result.score = rrf_scores[doc_id]   # 更新为 RRF 融合分数
            fused_results.append(result)

        # 打印向量 / ES / 融合后 的重叠统计，方便调试
        vector_ids = {r.id for r in vector_results}
        es_ids = {r.id for r in es_results}
        overlap = len(vector_ids & es_ids)
        logger.debug(
            f"RRF 融合统计 - 向量:{len(vector_ids)} + ES:{len(es_ids)} "
            f"= 重叠:{overlap}, 融合后:{len(fused_results)} 个"
        )

        return fused_results

    def _apply_rerank(
        self,
        query: str,
        rough_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """
        执行 Rerank 精排，返回前 top_k 个结果（阶段4）

        Args:
            query:         用户原始查询文本
            rough_results: RRF 融合后的候选文档列表
            top_k:         最终返回数量上限

        Returns:
            List[SearchResult]: 最终精排后的结果列表
        """
        # 提取文档文本列表
        document_texts = [r.content for r in rough_results]

        # 调用 Rerank 服务：Cross-Encoder 逐对打分
        rerank_scores = rerank_service.rerank(
            query=query,
            documents=document_texts,
            top_n=None,
        )

        # 将 Rerank 分数写回对应的 SearchResult 对象
        for scored_item in rerank_scores:
            original_idx = scored_item["index"]
            rough_results[original_idx].rerank_score = scored_item["relevance_score"]

        # 处理未得到 Rerank 分数的情况（容错）
        for result in rough_results:
            if result.rerank_score is None:
                result.rerank_score = 0.0

        # 按 rerank_score 降序排序
        rough_results.sort(key=lambda x: x.rerank_score, reverse=True)

        # 截取前 top_k 个
        final_results = rough_results[:top_k]

        if final_results:
            logger.info(
                f"[阶段4] Rerank 精排完成 - 最终 {len(final_results)} 个文档, "
                f"最高分: {final_results[0].rerank_score:.4f}, "
                f"最低分: {final_results[-1].rerank_score:.4f}"
            )
        else:
            logger.warning("[阶段4] Rerank 精排结果为空")

        return final_results


# 全局单例 - 与项目中其他 service 保持一致
vector_search_service = VectorSearchService()
