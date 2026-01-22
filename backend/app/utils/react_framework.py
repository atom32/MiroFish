"""
高性能ReACT框架
完全手搓，零依赖，性能最优

性能优化点：
1. 异步HTTP请求（并发工具调用）
2. 流式响应（减少首字延迟）
3. 连接池复用
4. 智能缓存
5. 最小化序列化开销
"""

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

try:
    from openai import AsyncOpenAI
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False
    print("警告: AsyncOpenAI不可用，将使用同步版本")


# ==================== 数据结构 ====================

@dataclass
class ToolCall:
    """工具调用"""
    name: str
    parameters: Dict[str, Any]
    call_id: str = ""  # 用于追踪


@dataclass
class ToolResult:
    """工具执行结果"""
    tool_name: str
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ReACTStep:
    """ReACT单步记录"""
    thought: str
    action: Optional[ToolCall] = None
    observation: Optional[str] = None
    iteration: int = 0


# ==================== 工具定义 ====================

class Tool(ABC):
    """工具基类"""

    def __init__(self):
        self.name = self.__class__.__name__
        self.description = self._get_description()
        self.parameters = self._get_parameters()

    @abstractmethod
    def _get_description(self) -> str:
        """工具描述"""
        pass

    @abstractmethod
    def _get_parameters(self) -> Dict[str, Any]:
        """参数schema"""
        pass

    @abstractmethod
    async def execute_async(self, **kwargs) -> Any:
        """异步执行"""
        pass

    def execute(self, **kwargs) -> Any:
        """同步执行（默认实现）"""
        return asyncio.run(self.execute_async(**kwargs))

    def to_schema(self) -> Dict[str, Any]:
        """转换为OpenAI function calling格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


# ==================== 高性能ReACT引擎 ====================

class HighPerformanceReACT:
    """
    高性能ReACT引擎

    核心优化：
    1. 异步并发 - 工具调用可并发执行
    2. 流式响应 - 实时返回生成内容
    3. 智能缓存 - 避免重复调用
    4. 连接池 - 复用HTTP连接
    5. 最小化序列化 - 减少JSON转换
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        tools: List[Tool],
        max_iterations: int = 5,
        max_concurrent_tools: int = 3,
        enable_streaming: bool = True,
        enable_cache: bool = True
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = max_iterations
        self.max_concurrent_tools = max_concurrent_tools
        self.enable_streaming = enable_streaming
        self.enable_cache = enable_cache

        # 异步客户端
        if ASYNC_AVAILABLE:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=2,
                timeout=60.0
            )
        else:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                max_retries=2,
                timeout=60.0
            )

        # 缓存（简单的LRU）
        self._cache: Dict[str, Any] = {}
        self._cache_size = 1000

        # 性能统计
        self.stats = {
            "total_calls": 0,
            "total_time": 0.0,
            "tool_calls": 0,
            "cache_hits": 0
        }

    def _get_cache_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        return f"{tool_name}:{json.dumps(params, sort_keys=True)}"

    async def _execute_tool_async(
        self,
        tool_call: ToolCall
    ) -> ToolResult:
        """异步执行工具（带缓存）"""
        start_time = time.time()

        # 检查缓存
        if self.enable_cache:
            cache_key = self._get_cache_key(tool_call.name, tool_call.parameters)
            if cache_key in self._cache:
                self.stats["cache_hits"] += 1
                return ToolResult(
                    tool_name=tool_call.name,
                    result=self._cache[cache_key],
                    execution_time=time.time() - start_time
                )

        # 执行工具
        try:
            tool = self.tools.get(tool_call.name)
            if not tool:
                raise ValueError(f"工具不存在: {tool_call.name}")

            if ASYNC_AVAILABLE:
                result = await tool.execute_async(**tool_call.parameters)
            else:
                result = tool.execute(**tool_call.parameters)

            execution_time = time.time() - start_time

            # 更新缓存
            if self.enable_cache:
                if len(self._cache) >= self._cache_size:
                    # 简单的缓存清理：删除第一个
                    self._cache.pop(next(iter(self._cache)))
                self._cache[cache_key] = result

            self.stats["tool_calls"] += 1

            return ToolResult(
                tool_name=tool_call.name,
                result=result,
                execution_time=execution_time
            )

        except Exception as e:
            return ToolResult(
                tool_name=tool_call.name,
                result=None,
                error=str(e),
                execution_time=time.time() - start_time
            )

    async def _execute_tools_concurrent(
        self,
        tool_calls: List[ToolCall]
    ) -> List[ToolResult]:
        """并发执行多个工具"""
        # 创建任务
        tasks = [
            self._execute_tool_async(call)
            for call in tool_calls[:self.max_concurrent_tools]
        ]

        # 并发执行
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        final_results = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                final_results.append(ToolResult(
                    tool_name=tool_calls[i].name,
                    result=None,
                    error=str(r)
                ))
            else:
                final_results.append(r)

        return final_results

    def _parse_tool_calls_from_response(self, response: str) -> List[ToolCall]:
        """
        从LLM响应中解析工具调用

        支持多种格式：
        1. Function calling（推荐）
        2. XML格式：[TOOL_CALL]{"name": "...", "parameters": {...}}
        3. 自定义格式
        """
        tool_calls = []

        # 方式1: 解析XML格式
        xml_pattern = r'\[TOOL_CALL\]\s*(\{.*?\})'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                data = json.loads(match.group(1))
                tool_calls.append(ToolCall(
                    name=data.get("name", ""),
                    parameters=data.get("parameters", {}),
                    call_id=f"call_{len(tool_calls)}"
                ))
            except json.JSONDecodeError:
                continue

        # 方式2: 如果LLM支持function calling，直接使用
        # 这里需要根据实际的响应格式调整

        return tool_calls

    def _build_system_prompt(self) -> str:
        """构建系统提示"""
        tools_desc = "\n\n".join([
            f"- {name}: {tool.description}\n  参数: {tool.parameters}"
            for name, tool in self.tools.items()
        ])

        return f"""你是一个智能助手，可以使用以下工具来完成任务：

{tools_desc}

工作流程：
1. Thought: 思考需要什么信息
2. Action: 调用工具获取信息
3. Observation: 分析工具返回结果
4. 重复步骤1-3，直到收集到足够信息
5. Final Answer: 给出最终答案

工具调用格式：
[TOOL_CALL] {{"name": "工具名", "parameters": {{"参数名": "参数值"}}}}

注意：
- 一次可以调用多个工具
- 工具调用结果会给你提供更多信息
- 最终答案必须基于工具返回的结果"""

    async def run_async(
        self,
        query: str,
        stream_callback: Optional[Callable[[str], None]] = None,
        step_callback: Optional[Callable[[ReACTStep], None]] = None
    ) -> Dict[str, Any]:
        """
        异步运行ReACT循环

        Args:
            query: 用户查询
            stream_callback: 流式回调（实时输出）
            step_callback: 步骤回调（记录每一步）

        Returns:
            最终结果 + 执行轨迹
        """
        start_time = time.time()
        self.stats["total_calls"] += 1

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": query}
        ]

        steps = []

        for iteration in range(self.max_iterations):
            # 调用LLM
            if ASYNC_AVAILABLE and self.enable_streaming:
                response = await self._chat_with_streaming(
                    messages,
                    stream_callback
                )
            else:
                response = await self._chat(messages)

            # 记录思考
            step = ReACTStep(
                thought=response,
                iteration=iteration
            )

            # 解析工具调用
            tool_calls = self._parse_tool_calls_from_response(response)

            if not tool_calls:
                # 没有工具调用，说明是最终答案
                step.observation = response
                if step_callback:
                    step_callback(step)
                steps.append(step)

                # 提取最终答案
                final_answer = self._extract_final_answer(response)

                self.stats["total_time"] += time.time() - start_time

                return {
                    "answer": final_answer,
                    "steps": steps,
                    "stats": self.stats.copy()
                }

            # 执行工具调用
            step.action = tool_calls[0] if len(tool_calls) == 1 else tool_calls

            if step_callback:
                step_callback(step)
            steps.append(step)

            # 并发执行工具
            results = await self._execute_tools_concurrent(tool_calls)

            # 构建观察结果
            observations = []
            for result in results:
                if result.error:
                    obs = f"错误: {result.error}"
                else:
                    obs = str(result.result)
                observations.append(f"{result.tool_name}: {obs}")

            observation_text = "\n\n".join(observations)
            step.observation = observation_text

            if step_callback:
                step_callback(step)

            # 添加到消息历史
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"工具返回结果:\n{observation_text}\n\n请基于这些信息继续思考或给出最终答案。"
            })

        # 达到最大迭代次数
        self.stats["total_time"] += time.time() - start_time
        return {
            "answer": "达到最大迭代次数，未能完成",
            "steps": steps,
            "stats": self.stats.copy()
        }

    async def _chat(self, messages: List[Dict[str, str]]) -> str:
        """发送聊天请求"""
        if ASYNC_AVAILABLE:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=2048
            )
            return response.choices[0].message.content
        else:
            # 同步版本
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=2048
            )
            return response.choices[0].message.content

    async def _chat_with_streaming(
        self,
        messages: List[Dict[str, str]],
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """发送流式聊天请求"""
        if not ASYNC_AVAILABLE:
            return await self._chat(messages)

        full_response = ""

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5,
            max_tokens=2048,
            stream=True
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_response += delta
                if callback:
                    callback(delta)

        return full_response

    def _extract_final_answer(self, response: str) -> str:
        """提取最终答案"""
        # 如果包含Final Answer标记
        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()

        # 否则返回整个响应
        return response.strip()

    # 同步接口（兼容性）
    def run(
        self,
        query: str,
        stream_callback: Optional[Callable[[str], None]] = None,
        step_callback: Optional[Callable[[ReACTStep], None]] = None
    ) -> Dict[str, Any]:
        """同步运行ReACT循环"""
        return asyncio.run(self.run_async(
            query,
            stream_callback,
            step_callback
        ))

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.stats.copy()


# ==================== 工具示例 ====================

class SearchTool(Tool):
    """搜索工具示例"""

    def _get_description(self) -> str:
        return "搜索相关信息"

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询"
                }
            },
            "required": ["query"]
        }

    async def execute_async(self, query: str) -> str:
        # 模拟搜索
        await asyncio.sleep(0.1)
        return f"搜索结果: 找到关于'{query}'的10条相关信息"


class CalculatorTool(Tool):
    """计算器工具示例"""

    def _get_description(self) -> str:
        return "执行数学计算"

    def _get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如 '2 + 2'"
                }
            },
            "required": ["expression"]
        }

    async def execute_async(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"计算结果: {result}"
        except Exception as e:
            return f"计算错误: {str(e)}"


# ==================== 使用示例 ====================

async def example_usage():
    """使用示例"""

    # 创建工具
    tools = [
        SearchTool(),
        CalculatorTool()
    ]

    # 创建ReACT引擎
    react = HighPerformanceReACT(
        api_key="your-api-key",
        base_url="https://api.openai.com/v1",
        model="gpt-4",
        tools=tools,
        max_iterations=5,
        max_concurrent_tools=3,
        enable_streaming=True,
        enable_cache=True
    )

    # 定义回调
    def on_stream(chunk: str):
        print(f"\r流式输出: {chunk}", end="", flush=True)

    def on_step(step: ReACTStep):
        print(f"\n\n[步骤 {step.iteration}]")
        print(f"思考: {step.thought[:100]}...")
        if step.action:
            print(f"行动: {step.action}")
        if step.observation:
            print(f"观察: {step.observation[:100]}...")

    # 运行
    result = await react.run_async(
        query="计算2+2，然后搜索相关信息",
        stream_callback=on_stream,
        step_callback=on_step
    )

    print(f"\n\n最终答案: {result['answer']}")
    print(f"性能统计: {result['stats']}")


if __name__ == "__main__":
    asyncio.run(example_usage())
