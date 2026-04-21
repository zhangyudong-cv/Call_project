import math
import time
import uuid
from typing import List, Dict, Any

from loguru import logger
from langchain_core.documents import Document

from app.config import config
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
        记忆召回策略：基于 Milvus 的时间衰减向量检索
        
        计算公式 (L2 距离衰减): 
        Adjusted_Distance = Raw_Distance * exp(lambda * delta_t)
        其中 lambda = ln(2) / half_life
        
        Args:
            session_id: 会话 ID
            query: 当前查询文本
            k: 最终提取的数量
            
        Returns:
            List[Document]: 召回的语义相关且考虑时间新鲜度的记忆片段
        """
        try:
            client = milvus_manager.get_client()
            now = int(time.time())
            
            # --- 1. 向量相似度初筛 (扩大候选池到 k*3) ---
            query_vector = vector_embedding_service.embed_query(query)
            rough_k = k * 3
            search_res = client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                filter=f'session_id == "{session_id}"', 
                limit=rough_k,
                output_fields=["content", "created_at"]
            )
            
            # --- 2. 应用时间衰减重排 ---
            # 计算衰减系数 lambda (ln 2 / half_life)
            half_life = config.memory_decay_half_life
            decay_lambda = math.log(2) / half_life if half_life > 0 else 0
            
            scored_candidates = []
            for hits in search_res:
                for hit in hits:
                    entity = hit.get("entity", {})
                    created_at = entity.get("created_at", 0)
                    raw_distance = hit.get("distance", 0.0)
                    
                    # 时间差（秒）
                    delta_t = max(0, now - created_at)
                    
                    # 衰减后的距离 (L2 距离越大越不相关，所以通过 exp 放大旧记录的距离)
                    decay_factor = math.exp(decay_lambda * delta_t)
                    adjusted_distance = raw_distance * decay_factor
                    
                    scored_candidates.append({
                        "doc": Document(
                            page_content=entity.get("content", ""),
                            metadata={"type": "sim_memory", "created_at": created_at, "raw_dist": raw_distance}
                        ),
                        "adjusted_dist": adjusted_distance
                    })
            
            # 按调整后的距离升序排列 (由于 L2 距离越小越好)
            scored_candidates.sort(key=lambda x: x["adjusted_dist"])
            
            # 截取前 k 个
            final_candidates = [item["doc"] for item in scored_candidates[:k]]
            
            # --- 3. 最终按时间戳正序排列 (保证注入 Prompt 时的语序逻辑) ---
            final_candidates.sort(key=lambda x: x.metadata.get("created_at", 0))
            
            logger.info(f"[衰减记忆召回] session_id={session_id}, 初始候选={len(scored_candidates)}, 最终召回={len(final_candidates)}")
            return final_candidates
            
        except Exception as e:
            logger.error(f"召回长期记忆失败: {e}")
            return []


# 全局单例
chat_memory_service = ChatMemoryService()
