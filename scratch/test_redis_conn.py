import asyncio
import sys
import os

# 将项目根目录加入路径
sys.path.append(os.path.abspath("."))

from app.core.redis_client import redis_manager
from loguru import logger

async def manual_test():
    try:
        logger.info("开始测试 Redis 连接...")
        # 1. 测试初始化
        saver = await redis_manager.get_saver()
        logger.info("Checkpointer (Saver) 初始化成功")
        
        # 2. 测试 Ping
        client = redis_manager.get_client()
        pong = await client.ping()
        logger.info(f"Redis Ping 结果: {pong}")
        
        # 3. 测试写入
        await client.set("test_key", "hello_redis")
        val = await client.get("test_key")
        logger.info(f"Redis 读写测试成功: {val}")
        
        print("RESULT: SUCCESS")
    except Exception as e:
        logger.exception("Redis 连接测试失败")
        print(f"RESULT: FAILED, ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(manual_test())
