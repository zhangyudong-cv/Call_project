"""对话记忆服务模块 - 基于 Milvus 的长短期记忆增强"""

import time
import uuid
from typing import List, Dict, Any

from loguru import logger
from langchain_core.documents import Document

from app.core.milvus_client import milvus_manager
from app.services.vector_embedding_service import vector_embedding_service


class ChatMemoryService:
    """对话记忆服务 - 负责跨会话/长短期记忆的存储与召回"""

    def __init__(self):
        self.collection_name = "chat_memory"

    def save_memory(self, session_id: str, question: str, answer: str) -> str:
        """
        保存一轮问答对到记忆库
        
        Args:
            session_id: 会话 ID (物理隔离依据)
            question: 用户问题
            answer: 模型回答
            
        Returns:
            str: 存入的 ID
        """
        try:
            # 1. 向量化用户问题 (之后用于相似度检索)
            vector = vector_embedding_service.embed_query(question)
            
            # 2. 构造文本内容
            content = f"User: {question}\nAssistant: {answer}"
            
            doc_id = str(uuid.uuid4())
            row = {
                "id": doc_id,
                "session_id": session_id,
                "vector": vector,
                "content": content,
                "created_at": int(time.time())
            }
            
            # 3. 写入 Milvus
            client = milvus_manager.get_client()
            client.insert(
                collection_name=self.collection_name,
                data=[row]
            )
            
            logger.debug(f"[记忆存储] session_id={session_id}, doc_id={doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"保存对话记忆失败: {e}")
            return ""

    def recall_memory(self, session_id: str, query: str, k: int = 5) -> List[Document]:
        """
        混合召回策略：相似度检索 + 最近 10 条
        
        Args:
            session_id: 会话 ID
            query: 当前查询文本
            k: 每路提取的数量
            
        Returns:
            List[Document]: 召回的记忆片段列表
        """
        try:
            client = milvus_manager.get_client()
            
            # --- 路 1: 相似度召回 ---
            query_vector = vector_embedding_service.embed_query(query)
            search_res = client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                filter=f'session_id == "{session_id}"', # 强制隔离
                limit=k,
                output_fields=["content", "created_at"]
            )
            
            sim_docs = []
            for hits in search_res:
                for hit in hits:
                    entity = hit.get("entity", {})
                    sim_docs.append(Document(
                        page_content=entity.get("content", ""),
                        metadata={"type": "sim_memory", "created_at": entity.get("created_at")}
                    ))
            
            # --- 路 2: 最近 10 条召回 ---
            recent_res = client.query(
                collection_name=self.collection_name,
                filter=f'session_id == "{session_id}"',
                limit=10,
                order_by="created_at",
                descending=True,
                output_fields=["content", "created_at"]
            )
            
            recent_docs = []
            for item in recent_res:
                recent_docs.append(Document(
                    page_content=item.get("content", ""),
                    metadata={"type": "recent_memory", "created_at": item.get("created_at")}
                ))
            
            # --- 合并去重 (基于创建时间或内容) ---
            seen_content = set()
            combined_docs = []
            
            # 优先处理最近的消息，保证时序性
            for doc in recent_docs + sim_docs:
                if doc.page_content not in seen_content:
                    combined_docs.append(doc)
                    seen_content.add(doc.page_content)
            
            # 按时间戳正序排列（让 LLM 按照对话发生的先后顺序阅读）
            combined_docs.sort(key=lambda x: x.metadata.get("created_at", 0))
            
            logger.info(f"[记忆召回] session_id={session_id}, 召回数量={len(combined_docs)}")
            return combined_docs
            
        except Exception as e:
            logger.error(f"召回对话记忆失败: {e}")
            return []


# 全局单例
chat_memory_service = ChatMemoryService()
