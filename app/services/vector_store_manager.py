"""向量存储管理器 - 绕过 LangChain 封装，直接使用 pymilvus 原生驱动以确保稳定性"""

import time
import uuid
from typing import List

from langchain_core.documents import Document
from loguru import logger

from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


class VectorStoreManager:
    """向量存储管理器 (原生驱动版)"""

    def __init__(self):
        """初始化管理器"""
        self.collection_name = "biz"

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        原生方法：添加文档到向量存储
        
        Args:
            documents: 文档列表
            
        Returns:
            List[str]: 文档 ID 列表
        """
        try:
            if not documents:
                return []
                
            start_time = time.time()
            logger.info(f"原生模式：正在准备存入 {len(documents)} 个文档片段到 Milvus...")

            # 1. 提取内容并进行向量化
            texts = [doc.page_content for doc in documents]
            # 这里调用我们已经实现的 DashScope 服务获取向量
            embeddings = vector_embedding_service.embed_documents(texts)
            
            # 2. 准备插入数据
            insert_data = []
            ids = []
            for i, (doc, vector) in enumerate(zip(documents, embeddings)):
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                
                # 构造符合 Milvus Schema 的行数据
                row = {
                    "id": doc_id,
                    "vector": vector,
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                insert_data.append(row)

            # 3. 使用原生客户端插入
            client = milvus_manager.get_client()
            res = client.insert(
                collection_name=self.collection_name,
                data=insert_data
            )
            
            elapsed = time.time() - start_time
            logger.info(f"✓ 原生模式：存储完成，耗时: {elapsed:.2f}s, 插入 ID 数量: {len(ids)}")
            return ids

        except Exception as e:
            logger.error(f"原生插入失败: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """
        原生方法：删除指定文件的所有文档
        """
        try:
            client = milvus_manager.get_client()
            
            # 先检查集合是否为空
            stats = client.get_collection_stats(collection_name=self.collection_name)
            if stats.get("row_count", 0) == 0:
                logger.info(f"向量库当前为空，跳过清理步骤: {file_path}")
                return 0

            # 原生表达式删除
            expr = f'metadata["_source"] == "{file_path}"'
            logger.info(f"原生模式：正在清理旧数据: {expr}")
            
            # 注意：MilvusClient 的 delete 接口稍有不同
            res = client.delete(
                collection_name=self.collection_name,
                filter=expr
            )
            return len(res) if isinstance(res, list) else 0
            
        except Exception as e:
            logger.warning(f"原生删除操作警告 (可能无旧数据): {e}")
            return 0

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        原生方法：相似度搜索
        """
        try:
            # 1. 向量化查询词
            query_vector = vector_embedding_service.embed_query(query)
            
            # 2. 调用原生搜索
            client = milvus_manager.get_client()
            search_res = client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                limit=k,
                output_fields=["content", "metadata"]
            )
            
            # 3. 转换为 LangChain Document 对象供后续 Agent 使用
            docs = []
            if search_res and len(search_res) > 0:
                for hits in search_res:
                    for hit in hits:
                        # hit 是字典对象
                        entity = hit.get("entity", {})
                        doc = Document(
                            page_content=entity.get("content", ""),
                            metadata=entity.get("metadata", {})
                        )
                        docs.append(doc)
            
            return docs
        except Exception as e:
            logger.error(f"原生搜索失败: {e}")
            return []

    def get_vector_store(self):
        """兼容性接口：返回自身"""
        return self


# 全局单例
vector_store_manager = VectorStoreManager()
