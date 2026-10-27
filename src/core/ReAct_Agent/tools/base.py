"""
工具工厂模块 (Base)

提供统一的工具创建工厂函数 `create_tool`，用于简化 LangChain `StructuredTool` 的创建过程，
并自动封装日志记录和异常处理，确保所有工具调用行为一致、可观测。

主要功能：
    - 标准化工具创建：只需提供工具名称、描述、输入 Schema 和执行函数，即可获得一个完整的StructuredTool实例。
    - 自动日志记录：工具调用开始时记录工具名称和输入参数，成功或失败时分别记录相应日志，便于链路追踪和问题排查。
    - 异常统一处理：捕获工具执行过程中的所有异常，记录错误堆栈后重新抛出，保证上层调用能够感知错误。
"""

import functools
import logging
from typing import Any, Callable, Optional
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _abbreviate(kwargs: dict) -> dict:
    """日志用：过长的列表参数（如注入的已返回段落内容）只记录条数，避免整段文本刷屏；短列表（如实体名）原样记录"""
    return {
        k: (f"<{len(v)} items>" if isinstance(v, (list, tuple)) and len(str(v)) > 300 else v)
        for k, v in kwargs.items()
    }


#    """统一创建工具的工厂函数，封装日志和异常处理，返回 LangChain StructuredTool 实例。"""
def create_tool(
    *,
    name: str,  # 工具名称
    description: str,  # 自然语言描述的每个工具用途、适用场景、输入输出格式说明
    args_schema: type[BaseModel],  # type: ignore
    func: Callable,  # 工具被调用时实际执行的函数
    return_direct: bool = False,  # 工具执行后返回给LLM，不直接返回结果给用户
) -> StructuredTool:
    """统一创建工具的工厂函数，封装日志和异常处理"""
    @functools.wraps(func)
    def wrapper(**kwargs) -> Any:
        # 监控工具调用的名称和参数
        logger.info(f"调用工具 '{name}'，参数: {_abbreviate(kwargs)}")
        try:
            result = func(**kwargs)
            # 监控工具调用是否成功
            logger.info(f"工具 '{name}' 调用成功")
            return result
        except Exception as e:
            logger.error(f"工具 '{name}' 执行失败: {e}", exc_info=True)
            raise
    return StructuredTool.from_function(
        func=wrapper,
        name=name,
        description=description,
        args_schema=args_schema,
        return_direct=return_direct,
    )
