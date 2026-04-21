"""Redis 基础基础设施模块 - 提供 Redis Stack 连接与 LangGraph 检查点支持"""

import os
from typing import Optional
from loguru import logger
from redis.asyncio import Redis, from_url
from langgraph.checkpoint.redis import AsyncRedisSaver

from app.config import config


class RedisManager:
    """Redis 管理器 - 维护连接池与 LangGraph 检查点"""

    def __init__(self):
        self._redis_url = self._build_url()
        self._client: Optional[Redis] = None
        self._saver_cm: Optional[AsyncRedisSaver] = None
        self._saver: Optional[AsyncRedisSaver] = None

    def _build_url(self) -> str:
        """构建 Redis 连接字符串"""
        auth = ""
        if config.redis_password:
            auth = f":{config.redis_password}@"
        
        url = f"redis://{auth}{config.redis_host}:{config.redis_port}/{config.redis_db}"
        return url

    async def get_client(self) -> Redis:
        """获取异步 Redis 客户端单例"""
        if self._client is None:
            logger.info(f"正在建立 Redis 连接: {config.redis_host}:{config.redis_port}")
            self._client = from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20
            )
            # 验证连接
            await self._client.ping()
        return self._client

    async def get_saver(self) -> AsyncRedisSaver:
        """获取 LangGraph Redis 检查点单例"""
        if self._saver is None:
            logger.info("初始化 LangGraph RedisSaver (从连接字符串)...")
            # 在 0.4.0+ 版本中，from_conn_string 返回上下文管理器
            self._saver_cm = AsyncRedisSaver.from_conn_string(self._redis_url)
            # 手动进入上下文以获得真正的 saver 实例
            self._saver = await self._saver_cm.__aenter__()
            # 进行初始化设置（如创建索引等）
            await self._saver.setup()
        return self._saver

    async def close(self):
        """关闭所有连接"""
        if self._saver_cm:
            await self._saver_cm.__aexit__(None, None, None)
            self._saver_cm = None
            self._saver = None
        if self._client:
            await self._client.close()
            self._client = None
        logger.info("Redis 连接已安全关闭")


# 全局单例管理器
redis_manager = RedisManager()
