"""Elasticsearch BM25 检索服务

使用 ES 的原生 BM25 算法做关键词全文检索，
与 Milvus 向量检索形成多路召回机制。

ES 版本要求: 7.12.x
IK 分词器版本: 与 ES 版本保持一致
"""

from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch, helpers
from loguru import logger

from app.config import config


class EsSearchService:
    """
    Elasticsearch BM25 检索服务

    核心职责:
        1. 管理 ES 索引（自动创建/复用）
        2. 批量写入文档分片（与 Milvus 共用同一批 chunks + 同一个 doc_id）
        3. 按来源文件路径删除旧的分片（更新文档时先清理）
        4. BM25 关键词全文检索，返回标准 SearchResult 格式
    """

    # ES 索引 Mapping 定义
    # - content 使用 IK 分词器（ik_max_word 索引 / ik_smart 查询）
    # - source  关键词类型，用于精确删除
    # - doc_id  与 Milvus 中的 id 字段一一对应，RRF 融合时用于去重
    # - metadata 存储原始元数据，不参与全文检索
    _INDEX_MAPPING = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "ik_max_word_analyzer": {
                        "type": "custom",
                        "tokenizer": "ik_max_word"
                    },
                    "ik_smart_analyzer": {
                        "type": "custom",
                        "tokenizer": "ik_smart"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},        # 与 Milvus 的 id 相同，用于 RRF 去重
                "content": {
                    "type": "text",
                    "analyzer": "ik_max_word",         # 写入时：最细粒度拆分，提升召回率
                    "search_analyzer": "ik_smart"      # 查询时：最粗粒度，提升精确度
                },
                "source": {"type": "keyword"},         # 来源文件路径，用于精确删除
                "metadata": {"type": "object", "enabled": False}  # 原始元数据，不索引
            }
        }
    }

    def __init__(self) -> None:
        """初始化 ES 客户端并确保索引存在"""
        self._client: Optional[Elasticsearch] = None
        self._index = config.es_index
        self._connected = False

        try:
            self._client = Elasticsearch(
                hosts=[{"host": config.es_host, "port": config.es_port}],
                timeout=10,
                max_retries=3,
                retry_on_timeout=True,
            )
            # 检查连通性
            if self._client.ping():
                self._connected = True
                self._ensure_index()
                logger.info(
                    f"EsSearchService 初始化完成 - "
                    f"ES: {config.es_host}:{config.es_port}, "
                    f"Index: {self._index}"
                )
            else:
                logger.warning(
                    f"ES 连接失败 (ping 超时): {config.es_host}:{config.es_port}，"
                    f"BM25 召回路径将被跳过"
                )
        except Exception as e:
            logger.warning(f"ES 初始化失败: {e}，BM25 召回路径将被跳过")

    def _ensure_index(self) -> None:
        """
        检查索引是否存在，不存在则创建

        使用 IK 分词器的 Mapping，支持中文关键词精确匹配。
        """
        try:
            if not self._client.indices.exists(index=self._index):
                self._client.indices.create(
                    index=self._index,
                    body=self._INDEX_MAPPING
                )
                logger.info(f"ES 索引已创建: {self._index}")
            else:
                logger.info(f"ES 索引已存在，复用: {self._index}")
        except Exception as e:
            logger.error(f"确保 ES 索引失败: {e}")
            raise

    def add_documents(
        self,
        documents: List[Any],
        doc_ids: List[str],
    ) -> None:
        """
        批量写入文档分片到 ES

        重要：doc_ids 必须与 Milvus 中写入的 ID 列表一一对应，
        保证两个存储系统使用相同的 doc_id，从而在 RRF 融合时能正确去重。

        Args:
            documents: LangChain Document 对象列表（来自 document_splitter_service）
            doc_ids:   与 Milvus 中存储的 id 字段严格对应的 UUID 列表
        """
        if not self._connected:
            logger.warning("ES 未连接，跳过写入")
            return

        if not documents or not doc_ids:
            return

        try:
            # 构建 bulk 写入的 action 列表
            actions = []
            for doc, doc_id in zip(documents, doc_ids):
                # 安全地从 metadata 中提取来源路径
                source_path = doc.metadata.get("_source", "")

                actions.append({
                    "_index": self._index,
                    "_id": doc_id,          # 使用与 Milvus 相同的 UUID 作为 ES 文档 ID
                    "_source": {
                        "doc_id": doc_id,
                        "content": doc.page_content,
                        "source": source_path,
                        "metadata": doc.metadata,
                    }
                })

            # 执行 bulk 写入
            success_count, errors = helpers.bulk(
                self._client,
                actions,
                raise_on_error=False,
                stats_only=False,
            )

            if errors:
                logger.warning(f"ES bulk 写入部分失败: {len(errors)} 条错误")

            logger.info(
                f"ES 写入完成: 成功 {success_count} 条, "
                f"共 {len(documents)} 条分片"
            )

        except Exception as e:
            logger.error(f"ES 批量写入失败: {e}")
            # 不抛出异常，避免 ES 故障影响主流程（Milvus 写入已成功）

    def delete_by_source(self, file_path: str) -> int:
        """
        按来源文件路径删除 ES 中的所有对应分片

        Args:
            file_path: 来源文件的规范化路径（与 metadata._source 字段保持一致）

        Returns:
            int: 实际删除的文档数量
        """
        if not self._connected:
            logger.warning("ES 未连接，跳过删除")
            return 0

        try:
            response = self._client.delete_by_query(
                index=self._index,
                body={
                    "query": {
                        "term": {"source": file_path}
                    }
                },
                refresh=True,   # 立即刷新，确保后续写入前删除已生效
            )
            deleted = response.get("deleted", 0)
            if deleted > 0:
                logger.info(f"ES 删除旧分片: {deleted} 条 (来源: {file_path})")
            return deleted

        except Exception as e:
            logger.warning(f"ES 删除操作警告 (可能无旧数据): {e}")
            return 0

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        BM25 全文检索

        使用 ES 的 multi_match 查询，对 content 字段进行 BM25 打分。
        查询时使用 ik_smart 分析器（粗粒度），提升精确度。

        Args:
            query:  用户查询文本
            top_k:  最多返回的候选文档数量

        Returns:
            List[Dict]: 包含 doc_id / content / metadata / bm25_score 的列表，
                        按 BM25 分数降序排列
        """
        if not self._connected:
            logger.warning("ES 未连接，BM25 召回返回空列表")
            return []

        try:
            body = {
                "query": {
                    "match": {
                        "content": {
                            "query": query,
                            "operator": "or",   # OR 匹配，最大化召回率
                            "minimum_should_match": "30%"  # 至少匹配 30% 的词项
                        }
                    }
                },
                "size": top_k,
                "_source": ["doc_id", "content", "source", "metadata"]
            }

            response = self._client.search(index=self._index, body=body)
            hits = response.get("hits", {}).get("hits", [])

            results = []
            for hit in hits:
                src = hit.get("_source", {})
                results.append({
                    "doc_id": src.get("doc_id", hit["_id"]),
                    "content": src.get("content", ""),
                    "metadata": src.get("metadata", {}),
                    "bm25_score": hit.get("_score", 0.0),
                })

            logger.info(
                f"ES BM25 检索完成: 召回 {len(results)} 个候选文档 "
                f"(query='{query[:40]}...')"
            )
            return results

        except Exception as e:
            logger.error(f"ES BM25 检索失败: {e}")
            return []

    @property
    def is_available(self) -> bool:
        """返回 ES 是否可用"""
        return self._connected


# 全局单例 - 与项目其他 service 保持一致
es_search_service = EsSearchService()
