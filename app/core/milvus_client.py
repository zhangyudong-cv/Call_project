"""Milvus 客户端工厂模块"""

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
    MilvusException,
)

from app.config import config


class MilvusClientManager:
    """Milvus 客户端管理器"""

    # 常量定义
    COLLECTION_NAME: str = "biz"
    MEMORY_COLLECTION_NAME: str = "chat_memory"  # 对话记忆集合
    VECTOR_DIM: int = 1024  # 统一使用 1024 维
    ID_MAX_LENGTH: int = 100
    SESSION_ID_MAX_LENGTH: int = 100
    CONTENT_MAX_LENGTH: int = 8000
    DEFAULT_SHARD_NUMBER: int = 2

    def __init__(self) -> None:
        """初始化 Milvus 客户端管理器"""
        self._client: MilvusClient | None = None
        self._collection: Collection | None = None

    def connect(self) -> MilvusClient:
        """
        连接到 Milvus 服务器并初始化 collection

        Returns:
            MilvusClient: Milvus 客户端实例

        Raises:
            RuntimeError: 连接或初始化失败时抛出
        """
        try:
            logger.info(f"正在连接到 Milvus: {config.milvus_host}:{config.milvus_port}")

            # 建立连接
            connections.connect(
                alias="default",
                host=config.milvus_host,
                port=str(config.milvus_port),
                timeout=config.milvus_timeout / 1000,  # 转换为秒
            )

            # 创建客户端
            uri = f"http://{config.milvus_host}:{config.milvus_port}"
            self._client = MilvusClient(uri=uri)

            logger.info("成功连接到 Milvus")

            # 1. 初始化业务知识库集合
            self._ensure_biz_collection()
            
            # 2. 初始化对话记忆集合
            self._ensure_memory_collection()

            return self._client

        except MilvusException as e:
            logger.error(f"Milvus 操作失败: {e}")
            self.close()
            raise RuntimeError(f"Milvus 操作失败: {e}") from e
        except ConnectionError as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e

    def _ensure_biz_collection(self) -> None:
        """确保 biz 集合存在且 Schema 正确"""
        if not utility.has_collection(self.COLLECTION_NAME):
            logger.info(f"集合 '{self.COLLECTION_NAME}' 不存在，正在创建...")
            self._create_biz_collection()
        else:
            coll = Collection(self.COLLECTION_NAME)
            schema = coll.schema
            field_names = [f.name for f in schema.fields]
            
            # 维度检查
            vector_field = next((f for f in schema.fields if f.name == "vector"), None)
            existing_dim = vector_field.params.get('dim') if vector_field else None
            
            needs_recreate = False
            if existing_dim != self.VECTOR_DIM:
                logger.warning(f"维度不匹配: {existing_dim} != {self.VECTOR_DIM}")
                needs_recreate = True
            elif "created_at" in field_names:
                logger.warning(f"集合 '{self.COLLECTION_NAME}' 存在多余的时间戳字段，正在移除...")
                needs_recreate = True
                
            if needs_recreate:
                logger.info(f"正在重建集合 '{self.COLLECTION_NAME}'...")
                utility.drop_collection(self.COLLECTION_NAME)
                self._create_biz_collection()
            
            # 无论是否重建，确保加载
            if self._collection is None:
                self._collection = Collection(self.COLLECTION_NAME)
            self._collection.load()
            logger.info(f"集合 '{self.COLLECTION_NAME}' 验证通过并已加载")

    def _ensure_memory_collection(self) -> None:
        """确保 chat_memory 集合存在"""
        if not utility.has_collection(self.MEMORY_COLLECTION_NAME):
            logger.info(f"集合 '{self.MEMORY_COLLECTION_NAME}' 不存在，正在创建...")
            self._create_memory_collection()
        else:
            coll = Collection(self.MEMORY_COLLECTION_NAME)
            coll.load()
            logger.info(f"集合 '{self.MEMORY_COLLECTION_NAME}' 验证通过并已加载")

    def _create_biz_collection(self) -> None:
        """创建业务知识库集合"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=self.ID_MAX_LENGTH, is_primary=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.VECTOR_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=self.CONTENT_MAX_LENGTH),
            FieldSchema(name="metadata", dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields=fields, description="Business knowledge")
        self._collection = Collection(name=self.COLLECTION_NAME, schema=schema)
        
        # 创建索引
        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        self._collection.create_index(field_name="vector", index_params=index_params)
        self._collection.load()  # 新建后立即加载
        logger.info(f"成功为 '{self.COLLECTION_NAME}' 创建索引并加载")

    def _create_memory_collection(self) -> None:
        """创建对话记忆集合"""
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=self.ID_MAX_LENGTH, is_primary=True),
            FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=self.SESSION_ID_MAX_LENGTH),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.VECTOR_DIM),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=self.CONTENT_MAX_LENGTH),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields=fields, description="Chat memory storage")
        coll = Collection(name=self.MEMORY_COLLECTION_NAME, schema=schema)
        
        # 创建索引
        index_params = {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        coll.create_index(field_name="vector", index_params=index_params)
        coll.load()  # 新建后立即加载
        logger.info(f"成功为 '{self.MEMORY_COLLECTION_NAME}' 创建索引并加载")

    def _load_collection(self) -> None:
        """加载 collection 到内存"""
        if self._collection is None:
            self._collection = Collection(self.COLLECTION_NAME)

        # 检查 collection 是否已加载（兼容多版本）
        try:
            # 方法 1: 尝试使用 utility.load_state（新版本）
            load_state = utility.load_state(self.COLLECTION_NAME)
            # load_state 返回字符串或枚举，如 "Loaded" 或 "NotLoad"
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
        except AttributeError:
            # 方法 2: 直接尝试加载，捕获 "already loaded" 异常
            try:
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            except MilvusException as e:
                error_msg = str(e).lower()
                if "already loaded" in error_msg or "loaded" in error_msg:
                    logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
                else:
                    raise
        except Exception as e:
            logger.error(f"加载 collection 失败: {e}")
            raise

    def get_collection(self) -> Collection:
        """
        获取 collection 实例

        Returns:
            Collection: collection 实例

        Raises:
            RuntimeError: collection 未初始化时抛出
        """
        if self._collection is None:
            raise RuntimeError("Collection 未初始化，请先调用 connect()")
        return self._collection

    def get_client(self) -> MilvusClient:
        """
        获取原生 MilvusClient 实例

        Returns:
            MilvusClient: Milvus 客户端实例
        """
        if self._client is None:
            # 如果尚未连接，尝试连接
            return self.connect()
        return self._client

    def health_check(self) -> bool:
        """
        健康检查

        Returns:
            bool: True 表示健康，False 表示异常
        """
        try:
            if self._client is None:
                return False

            # 尝试列出 connections
            _ = connections.list_connections()
            return True

        except (MilvusException, ConnectionError) as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False
        except Exception as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False

    def close(self) -> None:
        """关闭连接"""
        errors = []
        
        try:
            if self._collection is not None:
                self._collection.release()
                self._collection = None
        except Exception as e:
            errors.append(f"释放 collection 失败: {e}")

        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception as e:
            errors.append(f"断开连接失败: {e}")

        self._client = None
        
        if errors:
            error_msg = "; ".join(errors)
            logger.error(f"关闭 Milvus 连接时出现错误: {error_msg}")
        else:
            logger.info("已关闭 Milvus 连接")

    def __enter__(self) -> "MilvusClientManager":
        """上下文管理器入口"""
        _ = self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object
    ) -> None:
        """上下文管理器退出"""
        self.close()


# 全局单例
milvus_manager = MilvusClientManager()
