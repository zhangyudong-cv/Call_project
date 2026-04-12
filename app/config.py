"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理
"""

from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 应用配置
    app_name: str = "SuperBizAgent"
    app_version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 9900

    # DashScope 配置
    dashscope_api_key: str = ""  # 默认空字符串，实际使用需从环境变量加载
    dashscope_model: str = "qwen-max"
    dashscope_embedding_model: str = "text-embedding-v4"  # v4 支持多种维度（默认 1024）
    rerank_model: str = "qwen3-rerank"  # 重排序模型

    # Milvus 配置
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_timeout: int = 10000  # 毫秒

    # RAG 配置
    rag_top_k: int = 3                  # 最终返回给 LLM 的文档片段数量上限
    rag_rough_top_k: int = 10           # 向量检索粗排阶段召回的候选数量
    rag_rerank_threshold: float = 0.75  # Rerank 相关性分数阈值，低于此值的文档将被过滤
    rag_model: str = "qwen-max"         # 使用快速响应模型，不带扩展思考

    # Elasticsearch 配置（BM25 关键词召回路径）
    es_host: str = "192.168.100.128"    # ES 服务地址（与 Milvus 同机）
    es_port: int = 9200                  # ES HTTP 端口
    es_index: str = "biz_knowledge"      # 知识库索引名称
    es_top_k: int = 10                   # BM25 召回候选文档数量

    # RRF 融合参数
    rrf_k: int = 60                      # RRF 平滑常数，防止头部排名权重过大（工业经验值）
    rrf_top: int = 10                    # RRF 融合后送入 Rerank 的最大候选数

    # 文档分块配置
    chunk_max_size: int = 800
    chunk_overlap: int = 100

    # MCP 服务配置
    mcp_cls_transport: str = "streamable-http"
    mcp_cls_url: str = "http://127.0.0.1:8003/mcp"
    mcp_monitor_transport: str = "streamable-http"
    mcp_monitor_url: str = "http://127.0.0.1:8004/mcp"

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置"""
        return {
            "cls": {
                "transport": self.mcp_cls_transport,
                "url": self.mcp_cls_url,
            },
            "monitor": {
                "transport": self.mcp_monitor_transport,
                "url": self.mcp_monitor_url,
            }
        }


# 全局配置实例
config = Settings()
