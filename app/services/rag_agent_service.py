"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

from typing import Annotated, Any, AsyncGenerator, Dict, Sequence

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict
from langchain_qwq import ChatQwen

from app.config import config
from app.tools import get_current_time, retrieve_knowledge
from app.agent.mcp_client import get_mcp_client_with_retry
from app.services.chat_memory_service import chat_memory_service  # 引入记忆服务
from app.core.redis_client import redis_manager  # 引入 Redis 管理器

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 注意：需要配置环境变量 DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 否则默认访问的是新加坡站点
# 同时也需要配置环境变量 DASHSCOPE_API_KEY=your_api_key


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 5 条消息（根据用户要求）
    - 当消息少于等于 6 条时，不做修剪

    Args:
        state: Agent 状态

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    messages = state["messages"]

    # 如果消息数量较少，无需修剪 (1 条 System + 5 条历史)
    if len(messages) <= 6:
        return None

    # 提取第一条系统消息
    first_msg = messages[0]

    # 保留最近的 5 条消息
    recent_messages = messages[-5:]

    # 构建新的消息列表
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史 (Redis 滑动窗口): {len(messages)} -> {len(new_messages)} 条")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(self, streaming: bool = True):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()


        self.model = ChatQwen(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=streaming,
        )

        # 定义基础工具
        self.tools = [retrieve_knowledge, get_current_time]

        # MCP 客户端（延迟初始化，使用全局管理）
        self.mcp_tools: list = []

        # 检查点（延迟初始化，异步加载 RedisSaver）
        self.checkpointer = None

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(f"RAG Agent 服务初始化完成 (ChatQwen), model={self.model_name}, streaming={streaming}")

    async def _initialize_agent(self):
        """异步初始化 Agent（包括 MCP 工具）"""
        if self._agent_initialized:
            return

        # 使用全局 MCP 客户端管理器（带重试拦截器）
        mcp_client = await get_mcp_client_with_retry()

        # 获取 MCP 工具
        mcp_tools = await mcp_client.get_tools()
        logger.info(f"成功加载 {len(mcp_tools)} 个 MCP 工具")

        # 将 MCP 工具添加到实例变量中
        self.mcp_tools = mcp_tools

        # 合并所有工具
        all_tools = self.tools + self.mcp_tools

        # 获取 RedisSaver
        self.checkpointer = await redis_manager.get_saver()

        self.agent = create_react_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
            state_modifier=trim_messages_middleware,
        )

        self._agent_initialized = True


        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self, memory_context: str = "") -> str:
        """
        构建系统提示词

        Args:
            memory_context: 召回的历史记忆上下文
        """
        from textwrap import dedent
        
        # 如果有记忆，构造记忆部分
        memory_section = ""
        if memory_context:
            memory_section = f"\n[历史对话片段]\n{memory_context}\n"

        return dedent(f"""
            你是一个专业的AI助手，能够使用多种工具来帮助用户解决问题。
            {memory_section}
            工作原则:
            1. 理解用户需求，选择合适的工具来完成任务
            2. 当需要获取实时信息或专业知识时，主动使用相关工具
            3. 基于工具返回的结果提供准确、专业的回答
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 保持友好、专业的语气
            - 回答简洁明了，重点突出
            - 基于事实，不编造信息
            - 如有不确定的地方，明确说明

            请根据用户的问题，灵活使用可用工具，提供高质量的帮助。
        """).strip()

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（非流式）: {question}")

            # 1. 召回历史记忆 (物理隔离)
            memories = chat_memory_service.recall_memory(session_id, question)
            memory_text = "\n".join([m.page_content for m in memories]) if memories else ""
            
            # 2. 构建消息列表（带记忆增强的系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self._build_system_prompt(memory_text)),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(f"[会话 {session_id}] Agent 调用了工具: {tool_names}")

                logger.info(f"[会话 {session_id}] RAG Agent 查询完成（非流式）")
                
                # 3. 异步保存当次对话到向量库
                chat_memory_service.save_memory(session_id, question, answer)
                
                return answer

            logger.warning(f"[会话 {session_id}] Agent 返回结果为空")
            return ""

        except Exception as e:
            # 暴露 Python 3.11+ ExceptionGroup 的子异常，找到真正错误根源
            if hasattr(e, 'exceptions'):
                for i, sub_e in enumerate(e.exceptions):
                    logger.error(f"[会话 {session_id}] 子异常 [{i+1}]: {type(sub_e).__name__}: {sub_e}", exc_info=sub_e)
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（非流式）: {e}", exc_info=True)
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()

            logger.info(f"[会话 {session_id}] RAG Agent 收到查询（流式）: {question}")

            # 1. 召回历史记忆 (基于相似度+最新10条)
            memories = chat_memory_service.recall_memory(session_id, question)
            memory_text = "\n".join([m.page_content for m in memories]) if memories else ""
            
            # 2. 构建消息列表
            messages = [
                SystemMessage(content=self._build_system_prompt(memory_text)),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            logger.info(f"[会话 {session_id}] RAG Agent 查询完成（流式）")
            
            # 3. 保存记忆（流式场景下此处暂作简化，实际可累积 content 后存储）
            # 注意：流式存储通常需要从 state 中提取完整的回复，在此演示保存逻辑
            yield {"type": "complete"}

        except Exception as e:
            logger.error(f"[会话 {session_id}] RAG Agent 查询失败（流式）: {e}")
            yield {
                "type": "error",
                "data": str(e)
            }
            raise

    async def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 Redis checkpointer 中异步读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表
        """
        try:
            config = {"configurable": {"thread_id": session_id}}
            
            # 使用 aget 异步获取
            checkpoint_tuple = await self.checkpointer.aget(config)
            
            if not checkpoint_tuple:
                return []
            
            checkpoint_data = checkpoint_tuple.get("checkpoint", {})
            
            # 从检查点中提取消息
            messages = checkpoint_data.get("channel_values", {}).get("messages", [])
            
            history = []
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    continue
                    
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                
                from datetime import datetime
                history.append({
                    "role": role,
                    "content": content,
                    "timestamp": getattr(msg, 'timestamp', datetime.now().isoformat())
                })
            
            logger.info(f"获取会话历史成功: {session_id}, 数量: {len(history)}")
            return history
            
        except Exception as e:
            logger.error(f"获取会话历史失败: {session_id}, 错误: {e}")
            return []

    async def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从 Redis checkpointer 中删除）
        """
        try:
            # RedisSaver 通常支持特定的删除逻辑
            logger.info(f"正在清理会话记录: {session_id}")
            # 这里的实现取决于具体的 AsyncRedisSaver 版本，暂先保持逻辑占位
            return True
            
        except Exception as e:
            logger.error(f"清空会话历史失败: {session_id}, 错误: {e}")
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            # MCP 客户端由全局管理器统一管理，无需手动清理
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 全局单例 - 启用流式输出
rag_agent_service = RagAgentService(streaming=True)


# 第一步：存储 (Storage) —— 记忆的家
# 当对话开始时，每一条消息（User 说的话和 AI 的回答）都会被序列化成 JSON。

# Redis Key: 通常格式为 checkpoint:{session_id}。
# 内容: 存储的是整个对话的对象字典，包含消息列表、Token 统计等状态。
# 第二步：加载 (Loading) —— 唤醒记忆
# 当你再次发送消息时，query 方法会被调用：

# 代码执行 self.agent.ainvoke(..., config={"configurable": {"thread_id": session_id}})。
# LangGraph 看到 thread_id，立刻去 Redis 执行 GET 命令，把该会话的历史记录全部下载到内存中。
# 第三步：触发 (Trigger) —— 哨兵检查
# 在你刚刚挂载的 create_react_agent 内部，有一个“前置哨兵”（即 state_modifier）：

# 在把记忆发给大模型（LLM）之前，系统会先跑一遍 trim_messages_middleware。
# 此时的状态：内存里有一份完整的、从 Redis 取回的消息列表（比如已经有 10 条了）。
# 第四步：修剪 (Trimming) —— 滑动窗口动作
# 这是最关键的一步，发生在 trim_messages_middleware 内部：

# 计数：计算消息总数（发现是 10 条）。
# 提取：
# 拿走第 0 条（System Prompt）。
# 拿走最后 5 条（User最近的对话）。
# 重构命令：构造一个特殊的指令：
# RemoveMessage(id=REMOVE_ALL_MESSAGES)：告诉系统“清空当前内存和 Redis 里的所有旧消息”。
# [first_msg, *recent_messages]：把筛选后的 6 条（1+5）放进去。
# 第五步：写回 (Saving) —— 记忆同步
# 当 AI 回答完你的问题后：

# LangGraph 会合并新的消息，并调用 Redis 的 SET 命令。
# 结果：Redis 里的旧的长记忆被覆盖，现在只剩下被修剪后的 6 条 + 刚刚生成的 1 条新对话。