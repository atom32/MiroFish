# MiroFish 项目设计文档

> **多智能体群体智能预测引擎** - 详细技术设计解析

---

## 文档目录

1. [项目概述](#1-项目概述)
2. [系统架构](#2-系统架构)
3. [核心模块设计](#3-核心模块设计)
4. [算法详解](#4-算法详解)
5. [数据流设计](#5-数据流设计)
6. [接口设计](#6-接口设计)
7. [存储设计](#7-存储设计)
8. [并发与性能](#8-并发与性能)
9. [安全设计](#9-安全设计)
10. [扩展与改进](#10-扩展与改进)

---

## 1. 项目概述

### 1.1 核心理念

MiroFish 是一个**基于多智能体技术的群体智能预测引擎**，通过构建平行数字世界来预测未来趋势。

```
现实世界种子信息
        ↓
  GraphRAG 知识图谱
        ↓
  多智能体模拟环境
        ↓
  报告 Agent 分析
        ↓
    预测报告
```

### 1.2 设计目标

| 目标 | 描述 |
|------|------|
| **通用性** | 支持舆情、金融、创意写作等多种场景 |
| **高保真** | Agent具备独立人格、长期记忆、行为逻辑 |
| **可干预** | 支持"上帝视角"动态注入变量 |
| **可交互** | 与模拟Agent深度对话，探索细节 |

### 1.3 技术栈全景图

```
┌─────────────────────────────────────────────────────────────┐
│                        前端层 (Vue 3)                        │
├─────────────────────────────────────────────────────────────┤
│                        后端层 (Flask)                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   API      │  │  Services  │  │   Utils    │            │
│  └────────────┘  └────────────┘  └────────────┘            │
├─────────────────────────────────────────────────────────────┤
│                    外部服务依赖                               │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │   Zep      │  │   OASIS    │  │   LLM      │            │
│  │  Cloud     │  │   模拟引擎  │  │   API      │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 系统架构

### 2.1 整体架构图

```
┌──────────────────┐
│   用户浏览器     │
└────────┬─────────┘
         │ HTTP/WebSocket
         ↓
┌─────────────────────────────────────────────────────────┐
│                     前端 (Vue 3)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ 图谱视图 │  │ 模拟监控 │  │ 报告查看 │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└───────────────────────┬─────────────────────────────────┘
                        │ REST API
                        ↓
┌─────────────────────────────────────────────────────────┐
│                   后端 (Flask)                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │              API Layer                          │   │
│  │  /api/graph  /api/simulation  /api/report       │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Service Layer                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
│  │  │ Graph    │ │Simulation│ │  Report  │       │   │
│  │  │ Builder  │ │ Manager  │ │  Agent   │       │   │
│  │  └──────────┘ └──────────┘ └──────────┘       │   │
│  └─────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────┐   │
│  │              Utils Layer                        │   │
│  │  LLMClient  FileParser  Logger  Retry          │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         ↓              ↓              ↓
    ┌─────────┐   ┌─────────┐   ┌─────────┐
    │  Zep    │   │  OASIS  │   │  文件   │
    │  Cloud  │   │  脚本   │   │  存储   │
    └─────────┘   └─────────┘   └─────────┘
```

### 2.2 模块职责划分

#### 2.2.1 API 层 (`backend/app/api/`)

**职责**：HTTP请求处理、参数校验、响应封装

```python
# graph.py - 图谱相关API
POST   /api/graph/build          # 构建图谱
GET    /api/graph/:id            # 获取图谱数据
DELETE /api/graph/:id            # 删除图谱

# simulation.py - 模拟相关API
POST   /api/simulation/create    # 创建模拟
POST   /api/simulation/prepare   # 准备模拟
POST   /api/simulation/start     # 启动模拟
POST   /api/simulation/stop      # 停止模拟
GET    /api/simulation/:id       # 获取模拟状态
GET    /api/simulation/:id/actions  # 获取动作历史

# report.py - 报告相关API
POST   /api/report/generate      # 生成报告
GET    /api/report/:id           # 获取报告
GET    /api/report/:id/progress  # 获取生成进度
GET    /api/report/:id/log       # 获取生成日志
```

#### 2.2.2 Service 层 (`backend/app/services/`)

**核心服务矩阵**：

| 服务 | 文件 | 职责 |
|------|------|------|
| 图谱构建 | `graph_builder.py` | 调用Zep API构建知识图谱 |
| 实体读取 | `zep_entity_reader.py` | 从图谱读取并过滤实体 |
| 人设生成 | `oasis_profile_generator.py` | 为实体生成Agent人设 |
| 配置生成 | `simulation_config_generator.py` | LLM生成模拟配置参数 |
| 模拟管理 | `simulation_manager.py` | 模拟生命周期管理 |
| 模拟运行 | `simulation_runner.py` | 启动/停止/监控模拟进程 |
| 记忆更新 | `zep_graph_memory_updater.py` | 动态更新图谱记忆 |
| IPC通信 | `simulation_ipc.py` | 进程间通信（Interview） |
| 报告生成 | `report_agent.py` | ReACT模式生成报告 |
| 检索工具 | `zep_tools.py` | 图谱检索工具集 |

---

## 3. 核心模块设计

### 3.1 GraphRAG 构建模块

#### 3.1.1 设计思路

```
输入文本 → 分块 → 本体定义 → Zep处理 → 知识图谱
                ↓
         动态Schema生成
```

#### 3.1.2 类结构设计

```python
class GraphBuilderService:
    """
    图谱构建服务
    核心职责：
    1. 创建Zep图谱实例
    2. 设置本体（动态Schema）
    3. 批量添加文本数据
    4. 轮询等待处理完成
    5. 获取图谱统计信息
    """

    def __init__(self, api_key: str):
        self.client = Zep(api_key=api_key)
        self.task_manager = TaskManager()

    # 异步构建入口
    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3
    ) -> str:
        """返回task_id，后台线程执行"""

    # 核心工作流
    def _build_graph_worker(self, ...):
        """工作线程执行流程：
        1. create_graph() - 创建图谱
        2. set_ontology() - 设置本体
        3. add_text_batches() - 批量添加文本
        4. _wait_for_episodes() - 等待处理完成
        5. _get_graph_info() - 获取统计信息
        """

    # 本体设置（动态类型创建）
    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """动态生成Pydantic实体类和边类"""
```

#### 3.1.3 关键实现：动态本体生成

```python
def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
    """
    问题：Zep需要具体的Python类作为Schema
    方案：动态创建类

    输入：
    {
        "entity_types": [
            {
                "name": "Person",
                "attributes": [
                    {"name": "age", "description": "年龄"},
                    {"name": "gender", "description": "性别"}
                ]
            }
        ]
    }

    输出：
    class Person(EntityModel):
        __doc__ = "A Person entity."
        entity_age: Optional[EntityText] = Field(description="年龄", default=None)
        entity_gender: Optional[EntityText] = Field(description="性别", default=None)
    """

    # 1. 遍历实体类型定义
    for entity_def in ontology.get("entity_types", []):
        name = entity_def["name"]

        # 2. 构建属性字典
        attrs = {"__doc__": description}
        annotations = {}

        for attr_def in entity_def.get("attributes", []):
            attr_name = safe_attr_name(attr_def["name"])  # 避免保留字冲突
            attrs[attr_name] = Field(description=attr_desc, default=None)
            annotations[attr_name] = Optional[EntityText]

        attrs["__annotations__"] = annotations

        # 3. 动态创建类
        entity_class = type(name, (EntityModel,), attrs)
        entity_types[name] = entity_class

    # 4. 调用Zep API设置本体
    self.client.graph.set_ontology(
        graph_ids=[graph_id],
        entities=entity_types,
        edges=edge_definitions
    )
```

#### 3.1.4 批处理与状态管理

```python
def add_text_batches(
    self,
    graph_id: str,
    chunks: List[str],
    batch_size: int = 3,
    progress_callback: Optional[Callable] = None
) -> List[str]:
    """
    批量发送文本块

    设计考虑：
    1. 避免API限流 - 分批发送
    2. 可追溯性 - 收集episode uuid
    3. 进度反馈 - 回调函数
    4. 错误隔离 - 单批次失败不影响整体
    """

    episode_uuids = []

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i:i + batch_size]

        # 构建EpisodeData
        episodes = [
            EpisodeData(data=chunk, type="text")
            for chunk in batch_chunks
        ]

        # 发送到Zep
        batch_result = self.client.graph.add_batch(
            graph_id=graph_id,
            episodes=episodes
        )

        # 收集uuid用于后续状态追踪
        for ep in batch_result:
            ep_uuid = getattr(ep, 'uuid_', None) or getattr(ep, 'uuid', None)
            if ep_uuid:
                episode_uuids.append(ep_uuid)

        # 限流：每批次间隔1秒
        time.sleep(1)

    return episode_uuids
```

---

### 3.2 模拟管理模块

#### 3.2.1 状态机设计

```python
class SimulationStatus(str, Enum):
    """模拟状态流转"""
    CREATED     # 创建
    PREPARING   # 准备中（读取实体、生成人设、生成配置）
    READY       # 准备就绪
    RUNNING     # 运行中
    PAUSED      # 暂停
    STOPPED     # 已停止（手动）
    COMPLETED   # 已完成（自然结束）
    FAILED      # 失败

# 状态流转图：
# CREATED → PREPARING → READY → RUNNING → COMPLETED
#                                  ↘ PAUSED → RUNNING
#                                  ↘ STOPPED
#              ↘ FAILED (任意阶段)
```

#### 3.2.2 模拟准备流程

```python
def prepare_simulation(
    self,
    simulation_id: str,
    simulation_requirement: str,
    document_text: str,
    defined_entity_types: Optional[List[str]] = None,
    use_llm_for_profiles: bool = True,
    progress_callback: Optional[callable] = None,
    parallel_profile_count: int = 3
) -> SimulationState:
    """
    准备模拟环境（全程自动化）

    流程：
    ┌─────────────────────────────────────────────────────────┐
    │ 阶段1: 读取并过滤实体                                     │
    │   - 连接Zep图谱                                         │
    │   - 读取所有节点                                         │
    │   - 按类型过滤                                           │
    │   - 丰富边信息                                           │
    └─────────────────────────────────────────────────────────┘
                           ↓
    ┌─────────────────────────────────────────────────────────┐
    │ 阶段2: 生成Agent Profile（支持并行）                      │
    │   - 为每个实体生成人设                                   │
    │   - 基础模板：从实体属性提取                              │
    │   - LLM增强：丰富人格细节                                │
    │   - 实时保存：边生成边写入文件                           │
    │   - 并行控制：3个线程同时生成                            │
    └─────────────────────────────────────────────────────────┘
                           ↓
    ┌─────────────────────────────────────────────────────────┐
    │ 阶段3: LLM智能生成模拟配置                                │
    │   - 分析模拟需求                                         │
    │   - 生成时间配置（总时长、每轮时长）                      │
    │   - 生成活跃度配置（发言频率、互动频率）                  │
    │   - 生成平台配置                                         │
    │   - 保存配置文件                                         │
    └─────────────────────────────────────────────────────────┘
                           ↓
                        状态：READY
    """
```

#### 3.2.3 数据结构设计

```python
@dataclass
class SimulationState:
    """模拟状态（持久化到 state.json）"""
    simulation_id: str
    project_id: str
    graph_id: str

    # 平台启用状态
    enable_twitter: bool = True
    enable_reddit: bool = True

    # 状态
    status: SimulationStatus = SimulationStatus.CREATED

    # 准备阶段数据
    entities_count: int = 0        # 过滤后的实体数
    profiles_count: int = 0        # 生成的Profile数
    entity_types: List[str] = field(default_factory=list)

    # 配置生成信息
    config_generated: bool = False
    config_reasoning: str = ""     # LLM的配置生成推理

    # 运行时数据
    current_round: int = 0
    twitter_status: str = "not_started"
    reddit_status: str = "not_started"

    # 时间戳
    created_at: str
    updated_at: str

    # 错误信息
    error: Optional[str] = None

    # 持久化
    def to_dict(self) -> Dict[str, Any]:
        """序列化到JSON"""
```

---

### 3.3 模拟运行模块

#### 3.3.1 进程管理架构

```
┌─────────────────────────────────────────────────────────┐
│              SimulationRunner (类级别)                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │          全局状态字典                            │   │
│  │  _run_states: Dict[str, SimulationRunState]     │   │
│  │  _processes: Dict[str, subprocess.Popen]        │   │
│  │  _action_queues: Dict[str, Queue]              │   │
│  │  _monitor_threads: Dict[str, Thread]           │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  start_simulation()                                      │
│      ↓                                                   │
│  启动子进程 (run_parallel_simulation.py)                 │
│      ↓                                                   │
│  启动监控线程 (_monitor_simulation)                       │
│      ↓                                                   │
│  实时解析 actions.jsonl                                  │
└─────────────────────────────────────────────────────────┘
```

#### 3.3.2 运行状态管理

```python
@dataclass
class SimulationRunState:
    """模拟运行状态（实时）"""

    # 基础信息
    simulation_id: str
    runner_status: RunnerStatus

    # 进度信息
    current_round: int = 0
    total_rounds: int = 0
    simulated_hours: int = 0
    total_simulation_hours: int = 0

    # 双平台独立状态
    twitter_current_round: int = 0
    reddit_current_round: int = 0
    twitter_simulated_hours: int = 0
    reddit_simulated_hours: int = 0

    # 平台运行状态
    twitter_running: bool = False
    reddit_running: bool = False
    twitter_completed: bool = False  # 通过simulation_end事件检测
    reddit_completed: bool = False

    # 动作统计
    twitter_actions_count: int = 0
    reddit_actions_count: int = 0

    # 最近动作（内存缓存，限制50条）
    recent_actions: List[AgentAction] = field(default_factory=list)
    max_recent_actions: int = 50

    # 时间戳
    started_at: Optional[str] = None
    updated_at: str
    completed_at: Optional[str] = None

    # 进程管理
    process_pid: Optional[int] = None

    # 添加动作（自动更新统计）
    def add_action(self, action: AgentAction):
        self.recent_actions.insert(0, action)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions = self.recent_actions[:self.max_recent_actions]

        if action.platform == "twitter":
            self.twitter_actions_count += 1
        else:
            self.reddit_actions_count += 1

        self.updated_at = datetime.now().isoformat()
```

#### 3.3.3 监控线程实现

```python
def _monitor_simulation(cls, simulation_id: str):
    """
    监控线程：实时解析动作日志

    日志文件结构：
    uploads/simulations/{simulation_id}/
    ├── twitter/actions.jsonl
    ├── reddit/actions.jsonl
    └── simulation.log

    日志格式：
    {"round": 1, "agent_id": 0, "agent_name": "Alice",
     "action_type": "CREATE_POST", "action_args": {...},
     "timestamp": "2025-01-22T10:00:00"}

    事件类型：
    - simulation_start: 模拟开始
    - round_start: 轮次开始
    - round_end: 轮次结束（含round, simulated_hours）
    - simulation_end: 模拟结束（含total_rounds, total_actions）
    """
    sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)

    twitter_actions_log = os.path.join(sim_dir, "twitter", "actions.jsonl")
    reddit_actions_log = os.path.join(sim_dir, "reddit", "actions.jsonl")

    twitter_position = 0
    reddit_position = 0

    while process.poll() is None:  # 进程仍在运行
        # 读取 Twitter 日志（增量）
        if os.path.exists(twitter_actions_log):
            twitter_position = cls._read_action_log(
                twitter_actions_log, twitter_position, state, "twitter"
            )

        # 读取 Reddit 日志（增量）
        if os.path.exists(reddit_actions_log):
            reddit_position = cls._read_action_log(
                reddit_actions_log, reddit_position, state, "reddit"
            )

        # 持久化状态
        cls._save_run_state(state)

        time.sleep(2)  # 每2秒检查一次
```

#### 3.3.4 日志解析与事件检测

```python
def _read_action_log(
    cls,
    log_path: str,
    position: int,
    state: SimulationRunState,
    platform: str
) -> int:
    """
    增量读取动作日志

    特殊事件处理：
    1. simulation_end - 标记平台完成
    2. round_end - 更新轮次和时间
    3. 普通动作 - 添加到recent_actions
    """
    with open(log_path, 'r', encoding='utf-8') as f:
        f.seek(position)
        for line in f:
            action_data = json.loads(line.strip())

            # 事件类型检测
            if "event_type" in action_data:
                event_type = action_data.get("event_type")

                # 模拟结束事件
                if event_type == "simulation_end":
                    if platform == "twitter":
                        state.twitter_completed = True
                        state.twitter_running = False
                    elif platform == "reddit":
                        state.reddit_completed = True
                        state.reddit_running = False

                    # 检查是否所有平台都完成
                    if cls._check_all_platforms_completed(state):
                        state.runner_status = RunnerStatus.COMPLETED
                        state.completed_at = datetime.now().isoformat()

                # 轮次结束事件
                elif event_type == "round_end":
                    round_num = action_data.get("round", 0)
                    simulated_hours = action_data.get("simulated_hours", 0)

                    # 更新平台独立状态
                    if platform == "twitter":
                        state.twitter_current_round = round_num
                        state.twitter_simulated_hours = simulated_hours
                    elif platform == "reddit":
                        state.reddit_current_round = round_num
                        state.reddit_simulated_hours = simulated_hours

                    # 更新全局状态（取最大值）
                    state.current_round = max(state.current_round, round_num)
                    state.simulated_hours = max(
                        state.simulated_hours,
                        max(state.twitter_simulated_hours, state.reddit_simulated_hours)
                    )

                continue  # 跳过事件，不作为动作处理

            # 普通Agent动作
            action = AgentAction(
                round_num=action_data.get("round", 0),
                timestamp=action_data.get("timestamp", datetime.now().isoformat()),
                platform=platform,
                agent_id=action_data.get("agent_id", 0),
                agent_name=action_data.get("agent_name", ""),
                action_type=action_data.get("action_type", ""),
                action_args=action_data.get("action_args", {}),
                result=action_data.get("result"),
                success=action_data.get("success", True),
            )
            state.add_action(action)

        return f.tell()  # 返回新的文件位置
```

---

### 3.4 Report Agent 模块

#### 3.4.1 ReACT 框架设计

```
┌─────────────────────────────────────────────────────────┐
│                   ReACT 循环                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐     ┌──────────────┐                 │
│  │   Thought    │ ──→ │   Action     │                 │
│  │ (需要什么信息) │     │ (调用工具)   │                 │
│  └──────────────┘     └──────────────┘                 │
│         ↑                    ↓                          │
│         │              ┌──────────────┐                 │
│         │              │ Observation  │                 │
│         │              │ (工具返回结果) │                 │
│         │              └──────────────┘                 │
│         │                    ↓                          │
│         │              信息足够？                        │
│         │               /    \                          │
│         │             否      是                         │
│         │             /        \                         │
│         └─────────────┐    ┌────┴────────────────┐       │
│                       │    │   Final Answer      │       │
│                       └────│   (生成章节内容)     │       │
│                            └─────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

#### 3.4.2 工具集设计

```python
class ReportAgent:
    """
    报告生成Agent - ReACT模式

    工具层级体系：
    ┌─────────────────────────────────────────────────┐
    │ insight_forge (深度洞察 - 最强)                 │
    │   - 自动问题分解                                │
    │   - 多维度检索                                  │
    │   - 语义搜索 + 实体分析 + 关系链                │
    └─────────────────────────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │ panorama_search (广度搜索)                      │
    │   - 全局视图                                    │
    │   - 含历史/过期内容                             │
    │   - 事件演变追踪                                │
    └─────────────────────────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │ quick_search (快速检索)                         │
    │   - 简单查询                                    │
    │   - 验证事实                                    │
    │   - 速度优先                                    │
    └─────────────────────────────────────────────────┘
                         ↓
    ┌─────────────────────────────────────────────────┐
    │ interview_agents (深度采访)                     │
    │   - 真实Agent采访                               │
    │   - 双平台并发                                  │
    │   - 多视角分析                                  │
    └─────────────────────────────────────────────────┘
    """

    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": "【深度洞察检索 - 最强大】\n..."
                "parameters": {
                    "query": "你想深入分析的问题或话题",
                    "report_context": "当前报告章节的上下文"
                },
                "priority": "high"
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": "【广度搜索 - 获取全貌】\n..."
                "parameters": {
                    "query": "搜索查询",
                    "include_expired": "是否包含过期内容"
                },
                "priority": "medium"
            },
            "quick_search": {
                "name": "quick_search",
                "description": "【简单搜索 - 快速】\n..."
                "parameters": {
                    "query": "搜索查询",
                    "limit": "返回结果数量"
                },
                "priority": "low"
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": "【深度采访 - 真实Agent】\n..."
                "parameters": {
                    "interview_topic": "采访主题",
                    "max_agents": "最多采访数量"
                },
                "priority": "high"
            }
        }
```

#### 3.4.3 章节生成流程

```python
def _generate_section_react(
    self,
    section: ReportSection,
    outline: ReportOutline,
    previous_sections: List[str],
    progress_callback: Optional[Callable] = None,
    section_index: int = 0
) -> str:
    """
    使用ReACT模式生成单个章节

    流程：
    ┌─────────────────────────────────────────────────────────┐
    │ 1. 构建System Prompt                                   │
    │    - 报告标题、摘要、模拟需求                            │
    │    - 当前章节描述                                       │
    │    - ReACT工作流程说明                                  │
    │    - 工具使用建议                                       │
    │    - 格式规范（禁止标题，用粗体）                        │
    └─────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────┐
    │ 2. 构建User Prompt                                     │
    │    - 已完成章节内容（避免重复）                          │
    │    - 当前章节任务                                       │
    │    - 强调必须调用工具                                    │
    └─────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────┐
    │ 3. ReACT循环（最多5轮）                                 │
    │    for iteration in range(5):                           │
    │        - 调用LLM获取响应                                 │
    │        - 检查是否有Final Answer                          │
    │        - 解析工具调用                                    │
    │        - 执行工具（记录日志）                            │
    │        - 将结果添加到上下文                              │
    │        - 检查工具调用次数（至少2次）                     │
    └─────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────┐
    │ 4. 提取最终答案                                        │
    │    - 从"Final Answer:"后提取内容                        │
    │    - 清理多余格式                                        │
    │    - 记录生成日志                                        │
    └─────────────────────────────────────────────────────────┘
                          ↓
                      返回章节内容
    """

    # System Prompt核心部分
    system_prompt = f"""你是一个「未来预测报告」的撰写专家。

报告标题: {outline.title}
报告摘要: {outline.summary}
预测场景: {self.simulation_requirement}
当前章节: {section.title}

【最重要的规则】
1. 必须调用工具观察模拟世界（每个章节至少2次）
2. 必须引用Agent的原始言行
3. 禁止使用你自己的知识

【ReACT工作流程】
1. Thought: [分析需要什么信息]
2. Action: [调用工具]
    <tool_call>
    {{"name": "工具名称", "parameters": {{"参数名": "参数值"}}}}
    </tool_call>
3. Observation: [分析工具返回结果]
4. 重复1-3，直到信息足够
5. Final Answer: [生成章节内容]

【格式规范】
- ❌ 禁止使用任何标题（#、##、###等）
- ✅ 使用**粗体**代替小节标题
- ✅ 使用 > 格式引用原文
"""

    # ReACT循环
    for iteration in range(5):
        response = self.llm.chat(messages, temperature=0.5, max_tokens=4096)

        # 检查Final Answer
        if "Final Answer:" in response:
            if tool_calls_count < 2:
                # 强制继续调用工具
                continue
            final_answer = response.split("Final Answer:")[-1].strip()
            return final_answer

        # 解析工具调用
        tool_calls = self._parse_tool_calls(response)

        if not tool_calls:
            # 提示需要调用工具
            continue

        # 执行工具
        for call in tool_calls:
            result = self._execute_tool(
                call["name"],
                call.get("parameters", {}),
                report_context=report_context
            )
            tool_calls_count += 1

        # 添加到上下文
        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role": "user",
            "content": f"Observation: {result}\n\n继续..."
        })
```

#### 3.4.4 分章节生成与持久化

```python
def generate_report(
    self,
    progress_callback: Optional[Callable] = None,
    report_id: Optional[str] = None
) -> Report:
    """
    生成完整报告（分章节实时输出）

    文件结构：
    reports/{report_id}/
    ├── meta.json          # 报告元信息
    ├── outline.json       # 报告大纲
    ├── progress.json      # 生成进度（实时更新）
    ├── section_01.md      # 第1章节
    ├── section_02.md      # 第2章节
    ├── ...
    ├── full_report.md     # 完整报告
    ├── agent_log.jsonl    # Agent动作日志
    └── console_log.txt    # 控制台日志

    流程：
    ┌─────────────────────────────────────────────────────────┐
    │ 初始化：创建文件夹、日志记录器                           │
    └─────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────┐
    │ 阶段1：规划大纲                                         │
    │   - 调用LLM分析模拟需求                                 │
    │   - 生成章节结构（2-5个主章节）                          │
    │   - 保存 outline.json                                   │
    └─────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────┐
    │ 阶段2：逐章节生成（实时保存）                            │
    │   for i, section in enumerate(sections):                │
    │       - 调用 _generate_section_react()                  │
    │       - 生成主章节内容                                   │
    │       - 生成子章节内容（如有）                           │
    │       - 合并保存到 section_XX.md                        │
    │       - 更新 progress.json                              │
    │       - 记录 agent_log.jsonl                            │
    └─────────────────────────────────────────────────────────┘
                          ↓
    ┌─────────────────────────────────────────────────────────┐
    │ 阶段3：组装完整报告                                     │
    │   - 读取所有 section_XX.md                              │
    │   - 组装 full_report.md                                 │
    │   - 后处理（清理重复标题）                               │
    │   - 更新 progress.json 为 completed                     │
    └─────────────────────────────────────────────────────────┘
    """

    # 初始化
    if not report_id:
        report_id = f"report_{uuid.uuid4().hex[:12]}"

    ReportManager._ensure_report_folder(report_id)

    # 初始化日志
    self.report_logger = ReportLogger(report_id)
    self.console_logger = ReportConsoleLogger(report_id)

    report = Report(
        report_id=report_id,
        simulation_id=self.simulation_id,
        graph_id=self.graph_id,
        status=ReportStatus.PENDING
    )

    # 阶段1：规划大纲
    outline = self.plan_outline(progress_callback=...)
    ReportManager.save_outline(report_id, outline)

    # 阶段2：逐章节生成
    for i, section in enumerate(outline.sections):
        # 生成主章节
        section_content = self._generate_section_react(
            section=section,
            outline=outline,
            previous_sections=generated_sections,
            section_index=i + 1
        )

        # 生成子章节
        subsection_contents = []
        for j, subsection in enumerate(section.subsections):
            sub_content = self._generate_section_react(
                section=subsection,
                outline=outline,
                previous_sections=generated_sections,
                section_index=(i + 1) * 100 + (j + 1)
            )
            subsection_contents.append((subsection.title, sub_content))

        # 合并保存
        ReportManager.save_section_with_subsections(
            report_id, i + 1, section, subsection_contents
        )

        # 更新进度
        ReportManager.update_progress(
            report_id, "generating", progress,
            f"章节 {section.title} 已完成"
        )

    # 阶段3：组装完整报告
    report.markdown_content = ReportManager.assemble_full_report(
        report_id, outline
    )
    report.status = ReportStatus.COMPLETED

    return report
```

---

## 4. 算法详解

### 4.1 GraphRAG 算法

#### 4.1.1 算法流程

```
输入：原始文本 + 本体定义
│
├─ 步骤1：文本分块
│   └─ 算法：固定大小 + 滑动窗口
│       chunk_size = 500
│       chunk_overlap = 50
│       chunks = []
│       for i in range(0, len(text), chunk_size - overlap):
│           chunks.append(text[i:i + chunk_size])
│
├─ 步骤2：本体设置
│   └─ 动态Schema生成
│       for entity_def in ontology["entity_types"]:
│           创建 Pydantic EntityModel 子类
│       for edge_def in ontology["edge_types"]:
│           创建 Pydantic EdgeModel 子类
│       调用 Zep.set_ontology(entities, edges)
│
├─ 步骤3：批量处理
│   └─ 批次发送
│       for batch in chunks(batch_size=3):
│           episodes = [EpisodeData(data=chunk) for chunk in batch]
│           result = zep.graph.add_batch(episodes)
│           收集 episode_uuid
│           sleep(1)  # 限流
│
├─ 步骤4：状态轮询
│   └─ 轮询等待
│       while pending_episodes:
│           for uuid in pending_episodes:
│               episode = zep.graph.episode.get(uuid)
│               if episode.processed:
│                   pending_episodes.remove(uuid)
│           sleep(3)
│
└─ 输出：知识图谱（节点 + 边）
```

#### 4.1.2 时间复杂度分析

| 步骤 | 复杂度 | 瓶颈 |
|------|--------|------|
| 文本分块 | O(n) | - |
| 本体设置 | O(e) | e = 实体类型数 |
| 批量处理 | O(n/b) * O(1) | b = batch_size |
| 状态轮询 | O(n) * O(p) | p = 轮询间隔 |

**优化方向**：用WebSocket事件代替轮询

---

### 4.2 OASIS 模拟算法

#### 4.2.1 Agent 人设生成算法

```python
class OasisProfileGenerator:
    """
    Agent Profile 生成器

    策略：基础模板 + LLM增强
    """

    def generate_profiles_from_entities(
        self,
        entities: List[Dict],
        use_llm: bool = True,
        parallel_count: int = 3,
        realtime_output_path: str = None
    ) -> List[OasisAgentProfile]:
        """
        批量生成Agent Profile

        算法：
        ┌─────────────────────────────────────────────────────────┐
        │ 1. 基础提取（从实体属性）                                │
        │    profile = {                                         │
        │        "agent_id": entity["uuid_"][-6:],               │
        │        "username": entity["name"],                     │
        │        "bio": 实体摘要,                                  │
        │        "age": 从属性提取,                               │
        │        "gender": 从属性提取,                            │
        │        ...                                              │
        │    }                                                   │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ 2. LLM增强（可选，并行处理）                             │
        │    with ThreadPoolExecutor(max_workers=parallel_count): │
        │        futures = [                                      │
        │            executor.submit(                              │
        │                self._llm_enhance_profile,               │
        │                entity, graph_id                         │
        │            )                                            │
        │            for entity in entities                       │
        │        ]                                                │
        │    for future in as_completed(futures):                │
        │        profile = future.result()                        │
        │        实时保存到 realtime_output_path                   │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ 3. 格式转换                                             │
        │    Reddit → JSON                                        │
        │    Twitter → CSV                                        │
        └─────────────────────────────────────────────────────────┘
        """
```

#### 4.2.2 双平台并行模拟

```python
# run_parallel_simulation.py
def run_parallel_simulation(config_path: str):
    """
    双平台并行模拟

    架构：
    ┌─────────────────────────────────────────────────────────┐
    │ 主进程                                                  │
    │   ↓                                                    │
    │   启动Twitter模拟进程 (独立数据库)                       │
    │   启动Reddit模拟进程 (独立数据库)                        │
    │   ↓                                                    │
    │   等待两个进程完成                                       │
    │   ↓                                                    │
    │   合并日志到统一目录                                     │
    └─────────────────────────────────────────────────────────┘

    日志合并：
    uploads/simulations/{simulation_id}/
    ├── twitter/
    │   ├── actions.jsonl      # Twitter动作
    │   └── twitter_simulation.db
    └── reddit/
        ├── actions.jsonl      # Reddit动作
        └── reddit_simulation.db

    统一监控：
    simulation_runner._monitor_simulation() 同时监听两个文件
    """
    # 读取配置
    with open(config_path) as f:
        config = json.load(f)

    # 启动Twitter进程
    twitter_proc = subprocess.Popen([
        sys.executable, "run_twitter_simulation.py",
        "--config", config_path
    ])

    # 启动Reddit进程
    reddit_proc = subprocess.Popen([
        sys.executable, "run_reddit_simulation.py",
        "--config", config_path
    ])

    # 等待完成
    twitter_proc.wait()
    reddit_proc.wait()
```

---

### 4.3 ReACT 报告生成算法

#### 4.3.1 算法伪代码

```
function GENERATE_SECTION(section, previous_sections):
    messages = [
        SYSTEM_PROMPT,
        USER_PROMPT(previous_sections, section)
    ]

    tool_calls_count = 0

    for iteration in range(MAX_ITERATIONS):
        # Thought + Action
        response = LLM_CHAT(messages, temperature=0.5)

        # 检查Final Answer
        if "Final Answer:" in response:
            if tool_calls_count < MIN_TOOL_CALLS:
                messages.append(USER_PROMPT("继续调用工具"))
                continue
            return EXTRACT_FINAL_ANSWER(response)

        # 解析工具调用
        tool_calls = PARSE_TOOL_CALLS(response)

        if not tool_calls:
            messages.append(USER_PROMPT("请调用工具"))
            continue

        # 执行工具
        for call in tool_calls:
            if tool_calls_count >= MAX_TOOL_CALLS:
                break

            result = EXECUTE_TOOL(call["name"], call["parameters"])
            tool_calls_count += 1

            # Observation
            messages.append(ASSISTANT_MESSAGE(response))
            messages.append(
                USER_MESSAGE(f"Observation: {result}\n\n继续...")
            )

    # 达到最大迭代，强制生成
    return LLM_CHAT(messages)[-1]
```

#### 4.3.2 工具调用解析

```python
def _parse_tool_calls(self, response: str) -> List[Dict]:
    """
    从LLM响应中解析工具调用

    支持格式：
    1. XML风格：
       <tool_call>
       {"name": "insight_forge", "parameters": {"query": "..."}}
       </tool_call>

    2. 函数调用风格：
       [TOOL_CALL] insight_forge(query="...")
    """
    tool_calls = []

    # 格式1: XML风格
    xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    for match in re.finditer(xml_pattern, response, re.DOTALL):
        try:
            call_data = json.loads(match.group(1))
            tool_calls.append(call_data)
        except json.JSONDecodeError:
            pass

    # 格式2: 函数调用风格
    func_pattern = r'\[TOOL_CALL\]\s*(\w+)\s*\((.*?)\)'
    for match in re.finditer(func_pattern, response, re.DOTALL):
        tool_name = match.group(1)
        params_str = match.group(2)

        # 解析参数
        params = {}
        for param_match in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']', params_str):
            params[param_match.group(1)] = param_match.group(2)

        tool_calls.append({"name": tool_name, "parameters": params})

    return tool_calls
```

---

## 5. 数据流设计

### 5.1 端到端数据流

```
┌─────────────────────────────────────────────────────────────┐
│                     用户上传文件                             │
│                    (PDF/Markdown/TXT)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 1. 文件解析与文本提取                         │
│   utils/file_parser.py → 提取纯文本                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              2. 本体生成（可选，LLM生成）                     │
│   services/ontology_generator.py                            │
│   → 分析文本 → 生成实体类型和关系类型定义                    │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 3. GraphRAG 构建                            │
│   services/graph_builder.py                                 │
│   → Zep Cloud API → 知识图谱 (graph_id)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              4. 模拟准备（全自动）                           │
│   services/simulation_manager.py                            │
│   ├─ 读取实体 → zep_entity_reader.py                        │
│   ├─ 生成人设 → oasis_profile_generator.py                 │
│   └─ 生成配置 → simulation_config_generator.py             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                5. OASIS 模拟运行                             │
│   services/simulation_runner.py                             │
│   → 启动子进程 → Twitter/Reddit 并行                        │
│   → 实时监控 actions.jsonl                                  │
│   → 更新 run_state.json                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                6. Report Agent 分析                         │
│   services/report_agent.py                                  │
│   → 规划大纲 → 逐章节 ReACT 生成                            │
│   → 调用检索工具 (zep_tools.py)                             │
│   → 分章节保存 → 组装完整报告                               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                      7. 用户查看报告                         │
│   前端展示 Markdown + 交互式组件                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 关键数据结构

#### 5.2.1 知识图谱数据

```python
# Zep 图谱数据结构
{
    "graph_id": "mirofish_abc123...",
    "nodes": [
        {
            "uuid_": "uuid_1",
            "name": "张三",
            "labels": ["Person", "Student"],
            "summary": "武汉大学学生，计算机专业...",
            "attributes": {
                "entity_age": {"value": "20"},
                "entity_gender": {"value": "男"}
            },
            "created_at": "2025-01-22T10:00:00Z"
        }
    ],
    "edges": [
        {
            "uuid_": "edge_uuid_1",
            "name": "KNOWS",
            "fact": "张三认识李四",
            "fact_type": "KNOWS",
            "source_node_uuid": "uuid_1",
            "target_node_uuid": "uuid_2",
            "created_at": "2025-01-22T10:01:00Z",
            "valid_at": "2025-01-22T10:01:00Z",
            "episodes": ["ep_uuid_1"]
        }
    ]
}
```

#### 5.2.2 模拟配置数据

```python
# simulation_config.json
{
    "simulation_id": "sim_xyz789",
    "project_id": "proj_123",
    "graph_id": "mirofish_abc123",

    # 时间配置
    "time_config": {
        "total_simulation_hours": 72,
        "minutes_per_round": 30,
        "total_rounds": 144,
        "start_time": "2025-01-22T08:00:00"
    },

    # Agent配置
    "agent_configs": [
        {
            "agent_id": 0,
            "agent_name": "张三",
            "platform": "twitter",
            "personality": "活泼开朗，喜欢分享",
            "initial_posts": ["今天天气真好！"]
        }
    ],

    # 平台配置
    "platform_configs": {
        "twitter": {
            "enabled": true,
            "agent_count": 15
        },
        "reddit": {
            "enabled": true,
            "agent_count": 15
        }
    },

    # 配置生成推理
    "generation_reasoning": "根据舆情预测需求，生成72小时模拟..."
}
```

#### 5.2.3 动作日志数据

```python
# actions.jsonl (每行一个JSON)
{"round": 1, "timestamp": "2025-01-22T10:00:00", "platform": "twitter",
 "agent_id": 0, "agent_name": "张三", "action_type": "CREATE_POST",
 "action_args": {"content": "今天天气真好！"}}

{"event_type": "round_end", "round": 1, "simulated_hours": 0.5}

{"round": 2, "timestamp": "2025-01-22T10:30:00", "platform": "reddit",
 "agent_id": 1, "agent_name": "李四", "action_type": "COMMENT",
 "action_args": {"post_id": "post_1", "content": "确实不错！"}}

{"event_type": "simulation_end", "total_rounds": 144, "total_actions": 2150}
```

---

## 6. 接口设计

### 6.1 RESTful API 规范

#### 6.1.1 图谱管理 API

```python
# POST /api/graph/build
"""
构建知识图谱

Request:
{
    "text": "原始文本内容",
    "ontology": {
        "entity_types": [...],
        "edge_types": [...]
    },
    "graph_name": "My Graph"
}

Response:
{
    "task_id": "task_abc123",
    "status": "processing"
}
"""

# GET /api/graph/{graph_id}
"""
获取图谱数据

Response:
{
    "graph_id": "mirofish_abc123",
    "nodes": [...],
    "edges": [...],
    "node_count": 150,
    "edge_count": 320
}
"""
```

#### 6.1.2 模拟管理 API

```python
# POST /api/simulation/create
"""
创建模拟

Request:
{
    "project_id": "proj_123",
    "graph_id": "mirofish_abc123",
    "enable_twitter": true,
    "enable_reddit": true
}

Response:
{
    "simulation_id": "sim_xyz789",
    "status": "created"
}
"""

# POST /api/simulation/{simulation_id}/prepare
"""
准备模拟环境

Request:
{
    "simulation_requirement": "预测未来72小时的舆情走向",
    "document_text": "原始文档内容...",
    "defined_entity_types": ["Person", "Organization"],
    "use_llm_for_profiles": true,
    "parallel_profile_count": 3
}

Response (Server-Sent Events):
data: {"stage": "reading", "progress": 30, "message": "读取节点数据..."}
data: {"stage": "generating_profiles", "progress": 50, "message": "生成人设..."}
data: {"stage": "generating_config", "progress": 80, "message": "生成配置..."}
data: {"stage": "completed", "progress": 100, "message": "准备完成"}
"""

# POST /api/simulation/{simulation_id}/start
"""
启动模拟

Request:
{
    "platform": "parallel",  // "twitter" | "reddit" | "parallel"
    "max_rounds": 144,
    "enable_graph_memory_update": false
}

Response:
{
    "simulation_id": "sim_xyz789",
    "status": "running",
    "process_pid": 12345
}
"""

# GET /api/simulation/{simulation_id}/status
"""
获取模拟状态（实时）

Response:
{
    "simulation_id": "sim_xyz789",
    "status": "running",
    "current_round": 72,
    "total_rounds": 144,
    "progress_percent": 50.0,
    "twitter_current_round": 72,
    "reddit_current_round": 72,
    "twitter_running": true,
    "reddit_running": true,
    "total_actions_count": 1050,
    "recent_actions": [...]
}
"""

# GET /api/simulation/{simulation_id}/actions
"""
获取动作历史（分页）

Query Parameters:
- limit: 返回数量（默认100）
- offset: 偏移量（默认0）
- platform: 平台过滤（可选）
- agent_id: Agent过滤（可选）

Response:
{
    "actions": [
        {
            "round_num": 72,
            "timestamp": "2025-01-22T22:00:00",
            "platform": "twitter",
            "agent_id": 5,
            "agent_name": "王五",
            "action_type": "CREATE_POST",
            "action_args": {"content": "..."}
        }
    ],
    "total_count": 1050,
    "has_more": true
}
"""
```

#### 6.1.3 报告生成 API

```python
# POST /api/report/generate
"""
生成报告

Request:
{
    "simulation_id": "sim_xyz789",
    "graph_id": "mirofish_abc123",
    "simulation_requirement": "预测未来72小时的舆情走向"
}

Response:
{
    "report_id": "report_def456",
    "status": "planning"
}
"""

# GET /api/report/{report_id}/progress
"""
获取报告生成进度（实时）

Response:
{
    "status": "generating",
    "progress": 65,
    "message": "正在生成章节: 人群行为预测分析 (3/5)",
    "current_section": "人群行为预测分析",
    "completed_sections": [
        "预测场景与核心发现",
        "舆论传播态势分析"
    ],
    "updated_at": "2025-01-22T23:30:00"
}
"""

# GET /api/report/{report_id}
"""
获取报告

Response:
{
    "report_id": "report_def456",
    "simulation_id": "sim_xyz789",
    "status": "completed",
    "outline": {
        "title": "武汉大学舆情预测报告",
        "summary": "...",
        "sections": [...]
    },
    "markdown_content": "# 报告标题\n...",
    "created_at": "2025-01-22T23:00:00",
    "completed_at": "2025-01-23T00:00:00"
}
"""

# GET /api/report/{report_id}/log
"""
获取Agent生成日志（流式）

Query Parameters:
- from_line: 起始行号（增量获取）

Response:
{
    "logs": [
        {
            "timestamp": "2025-01-22T23:00:00",
            "action": "section_start",
            "stage": "generating",
            "section_title": "预测场景与核心发现",
            "details": {"message": "开始生成章节"}
        },
        ...
    ],
    "total_lines": 1500,
    "from_line": 0,
    "has_more": false
}
"""
```

### 6.2 Interview 机制

```python
# IPC通信协议（命名管道/Unix Socket）

# 1. 检查环境存活
GET /api/simulation/{simulation_id}/env/alive
Response: {"alive": true}

# 2. 采访单个Agent
POST /api/simulation/{simulation_id}/interview
Request:
{
    "agent_id": 5,
    "prompt": "你对最近的舆情事件有什么看法？",
    "platform": null,  // null表示双平台都采访
    "timeout": 60
}

Response:
{
    "success": true,
    "agent_id": 5,
    "result": {
        "twitter_response": "我认为...",
        "reddit_response": "我的看法是..."
    },
    "timestamp": "2025-01-22T23:00:00"
}

# 3. 批量采访
POST /api/simulation/{simulation_id}/interview/batch
Request:
{
    "interviews": [
        {"agent_id": 5, "prompt": "..."},
        {"agent_id": 8, "prompt": "..."}
    ],
    "timeout": 120
}
```

---

## 7. 存储设计

### 7.1 文件系统布局

```
uploads/
├── projects/              # 项目数据
│   └── {project_id}/
│       ├── meta.json
│       └── files/
│           └── original.pdf
│
├── simulations/           # 模拟数据
│   └── {simulation_id}/
│       ├── state.json              # 模拟状态
│       ├── simulation_config.json   # 模拟配置
│       ├── run_state.json           # 运行状态
│       ├── progress.json            # 生成进度
│       ├── reddit_profiles.json     # Reddit人设
│       ├── twitter_profiles.csv     # Twitter人设
│       ├── twitter/
│       │   ├── actions.jsonl        # Twitter动作日志
│       │   └── twitter_simulation.db
│       ├── reddit/
│       │   ├── actions.jsonl        # Reddit动作日志
│       │   └── reddit_simulation.db
│       └── simulation.log           # 主进程日志
│
└── reports/               # 报告数据
    └── {report_id}/
        ├── meta.json              # 报告元信息
        ├── outline.json           # 报告大纲
        ├── progress.json          # 生成进度
        ├── section_01.md          # 第1章节
        ├── section_02.md          # 第2章节
        ├── ...
        ├── full_report.md         # 完整报告
        ├── agent_log.jsonl        # Agent动作日志
        └── console_log.txt        # 控制台日志
```

### 7.2 数据库设计

#### 7.2.1 OASIS SQLite 数据库

```sql
-- Twitter/Reddit 模拟数据库结构

-- 用户表（Agent）
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    user_name TEXT NOT NULL,
    user_description TEXT,
    platform TEXT  -- 'twitter' or 'reddit'
);

-- 内容表（帖子/评论）
CREATE TABLE contents (
    content_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    content TEXT NOT NULL,
    platform TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- 互动表（点赞/评论/关注）
CREATE TABLE interactions (
    interaction_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    target_user_id INTEGER,
    interaction_type TEXT,  -- 'LIKE', 'COMMENT', 'FOLLOW'
    platform TEXT,
    created_at TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (target_user_id) REFERENCES users(user_id)
);

-- 追踪表（Interview记录）
CREATE TABLE trace (
    trace_id INTEGER PRIMARY KEY,
    user_id INTEGER,
    action TEXT,  -- 'CREATE_POST', 'LIKE_POST', 'interview', etc.
    info TEXT,   -- JSON格式的详细信息
    created_at TIMESTAMP
);
```

#### 7.2.2 Zep Cloud 图数据库

```
节点类型：
- Entity: 基础实体
- Person: 人物
- Organization: 组织
- Location: 地点
- Event: 事件
...

边类型：
- KNOWS: 认识关系
- PART_OF: 属于关系
- LOCATED_AT: 位于关系
- HAPPENED_AT: 发生于关系
...

属性：
- 节点属性：name, summary, created_at
- 边属性：fact, fact_type, valid_at, invalid_at, episodes
```

---

## 8. 并发与性能

### 8.1 并发模型

```
┌─────────────────────────────────────────────────────────────┐
│                    Flask 主进程                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              API 请求处理线程池                     │  │
│   │   Thread-1: /api/graph/build                        │  │
│   │   Thread-2: /api/simulation/start                   │  │
│   │   Thread-3: /api/report/generate                    │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                            │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              后台工作线程                             │  │
│   │   Thread-4: 图谱构建工作线程                          │  │
│   │   Thread-5: 报告生成工作线程                          │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘

                         │
                         ↓ 启动子进程
┌─────────────────────────────────────────────────────────────┐
│              OASIS 模拟子进程（独立）                        │
│   ┌─────────────────────────────────────────────────────┐  │
│   │          Twitter 模拟进程                            │  │
│   │   - SQLite: twitter_simulation.db                   │  │
│   │   - 日志: twitter/actions.jsonl                     │  │
│   └─────────────────────────────────────────────────────┘  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │          Reddit 模拟进程                             │  │
│   │   - SQLite: reddit_simulation.db                    │  │
│   │   - 日志: reddit/actions.jsonl                      │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 性能优化策略

| 策略 | 当前实现 | 优化方向 |
|------|----------|----------|
| **批处理** | 图谱构建batch_size=3 | 动态调整批次大小 |
| **轮询** | 固定3秒间隔 | WebSocket事件通知 |
| **并行** | 人设生成3线程 | 可配置线程池 |
| **缓存** | 无 | 实体检索缓存 |
| **流式输出** | 分章节保存 | 实现SSE流式API |
| **日志管理** | 完整存储 | 日志轮转/压缩 |

### 8.3 资源消耗分析

```
典型模拟的资源消耗（30个Agent，144轮次）：

CPU:
- 主进程: 5-10%（API处理 + 监控）
- Twitter进程: 20-30%
- Reddit进程: 20-30%
- 峰值总计: ~60%

内存:
- 主进程: 200-500 MB（状态缓存）
- Twitter进程: 300-800 MB
- Reddit进程: 300-800 MB
- 峰值总计: ~2 GB

存储:
- SQLite数据库: 50-100 MB/平台
- 日志文件: 10-50 MB
- 总计: ~200 MB/模拟

网络:
- Zep API调用: ~1000次/构建
- LLM API调用: ~50-100次/报告
- 峰值带宽: ~1 Mbps
```

---

## 9. 安全设计

### 9.1 安全威胁分析

| 威胁 | 影响 | 现有防护 | 改进建议 |
|------|------|----------|----------|
| **API密钥泄露** | 高 | 环境变量 | 密钥轮换、审计日志 |
| **注入攻击** | 中 | 基础输入校验 | 参数化查询、深度验证 |
| **路径遍历** | 中 | os.path.join | 路径规范化、白名单 |
| **资源耗尽** | 中 | 无 | 请求限流、配额管理 |
| **并发竞态** | 低 | 无 | 线程锁、原子操作 |

### 9.2 输入验证框架

```python
# 建议添加的验证装饰器
from functools import wraps
from pydantic import BaseModel, ValidationError

def validate_request(model_class):
    """请求参数验证装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # 从Flask request获取数据
                data = request.get_json()
                validated_data = model_class(**data)
                return func(validated_data=validated_data.dict(), *args, **kwargs)
            except ValidationError as e:
                return jsonify({"error": str(e)}), 400
        return wrapper
    return decorator

# 使用示例
class BuildGraphRequest(BaseModel):
    text: str
    ontology: Dict[str, Any]
    graph_name: str = "MiroFish Graph"

    class Config:
        str_max_length = 1000000  # 限制文本长度

@app.route("/api/graph/build", methods=["POST"])
@validate_request(BuildGraphRequest)
def build_graph(validated_data):
    # validated_data 已经过验证
    pass
```

---

## 10. 扩展与改进

### 10.1 短期优化（1-2个月）

#### 优先级1：性能优化
```python
# 1. WebSocket 替代轮询
class ZepWebSocketClient:
    async def subscribe_episodes(self, episode_uuids):
        ws = await websockets.connect(ZEP_WS_URL)
        await ws.send(json.dumps({
            "action": "subscribe",
            "episodes": episode_uuids
        }))
        async for msg in ws:
            data = json.loads(msg)
            if data["status"] == "completed":
                # 处理完成事件
                pass

# 2. 缓存层
from functools import lru_cache
from cachetools import TTLCache

entity_cache = TTLCache(maxsize=1000, ttl=3600)

@lru_cache(maxsize=100)
def get_entity_summary(graph_id: str, entity_name: str):
    # 缓存实体摘要
    pass
```

#### 优先级2：并发安全
```python
import threading

class SimulationRunner:
    _lock = threading.RLock()

    @classmethod
    def _save_run_state(cls, state):
        with cls._lock:
            cls._run_states[state.simulation_id] = state
            # ... 持久化
```

#### 优先级3：配置外部化
```yaml
# config/config.yaml
graph:
  chunk_size: 500
  chunk_overlap: 50
  batch_size: 3

simulation:
  default_rounds: 144
  parallel_profile_count: 3

report:
  max_tool_calls_per_section: 5
  max_reflection_rounds: 3
```

### 10.2 中期扩展（3-6个月）

#### 1. 分布式架构
```
┌─────────────────────────────────────────────────────────┐
│              API Gateway (Nginx/Kong)                   │
└──────────────┬──────────────────────┬──────────────────┘
               │                      │
      ┌────────▼────────┐    ┌───────▼────────┐
      │  Flask 实例 1   │    │  Flask 实例 2 │
      └────────┬────────┘    └───────┬────────┘
               │                      │
               └──────────┬───────────┘
                          │
               ┌──────────▼──────────┐
               │   Redis Cache       │
               │   Message Queue     │
               └─────────────────────┘
                          │
      ┌───────────────────┼───────────────────┐
      │                   │                   │
┌─────▼─────┐    ┌───────▼──────┐    ┌─────▼─────┐
│ Worker 1  │    │  Worker 2    │    │ Worker 3  │
│ (图谱构建) │    │ (模拟运行)   │    │ (报告生成) │
└───────────┘    └──────────────┘    └───────────┘
```

#### 2. 微服务拆分
```
mirofish/
├── graph-service/      # 图谱构建服务
├── simulation-service/ # 模拟运行服务
├── report-service/     # 报告生成服务
├── api-gateway/        # API网关
└── shared/             # 共享库
```

#### 3. 消息队列集成
```python
# 使用Celery进行异步任务
from celery import Celery

app = Celery('mirofish', broker='redis://localhost:6379')

@app.task
def build_graph_async(text, ontology):
    graph_builder = GraphBuilderService()
    return graph_builder.build_graph_async(text, ontology)

@app.task
def generate_report_async(simulation_id, graph_id):
    report_agent = ReportAgent(graph_id, simulation_id)
    return report_agent.generate_report()
```

### 10.3 长期愿景（6-12个月）

#### 1. 实时协作
```python
# WebSocket实时同步
class SimulationCollaboration:
    """
    多用户实时协作

    功能：
    - 广播模拟进度
    - 同步标注操作
    - 多人Interview
    """

    async def broadcast_action(self, simulation_id, action):
        await websocket_manager.broadcast({
            "type": "new_action",
            "simulation_id": simulation_id,
            "action": action.to_dict()
        })

    async def sync_annotation(self, simulation_id, user_id, annotation):
        # 同步用户标注
        pass
```

#### 2. 智能推荐
```python
class SimulationRecommender:
    """
    模拟参数智能推荐

    基于：
    - 历史模拟效果
    - 当前数据特征
    - 机器学习模型
    """

    def recommend_duration(self, graph_size, complexity):
        """推荐模拟时长"""
        if graph_size < 50:
            return 24  # 小规模：1天
        elif graph_size < 200:
            return 72  # 中规模：3天
        else:
            return 168  # 大规模：1周

    def recommend_agent_count(self, entity_types):
        """推荐Agent数量"""
        # 基于实体类型分布
        pass
```

#### 3. 自动化测试
```python
# 集成测试框架
import pytest

class TestSimulationE2E:
    """端到端测试"""

    def test_full_simulation_flow(self):
        # 1. 上传文件
        response = client.post("/api/graph/build", json={...})
        task_id = response.json["task_id"]

        # 2. 等待图谱构建
        graph_id = self.wait_for_task(task_id)

        # 3. 创建并准备模拟
        simulation = self.create_simulation(graph_id)
        self.prepare_simulation(simulation["id"])

        # 4. 运行模拟
        self.start_simulation(simulation["id"])
        self.wait_for_completion(simulation["id"])

        # 5. 生成报告
        report = self.generate_report(simulation["id"])

        # 6. 验证结果
        assert report["status"] == "completed"
        assert len(report["outline"]["sections"]) >= 2
```

---

## 附录

### A. 配置参数手册

| 参数 | 位置 | 默认值 | 说明 |
|------|------|--------|------|
| `chunk_size` | graph_builder.py | 500 | 文本分块大小 |
| `chunk_overlap` | graph_builder.py | 50 | 分块重叠大小 |
| `batch_size` | graph_builder.py | 3 | Zep API批次大小 |
| `max_rounds` | simulation_runner.py | None | 最大模拟轮数 |
| `parallel_profile_count` | simulation_manager.py | 3 | 并行人设生成数 |
| `MAX_TOOL_CALLS_PER_SECTION` | report_agent.py | 5 | 每章节最大工具调用数 |
| `MAX_REFLECTION_ROUNDS` | report_agent.py | 3 | 最大反思轮数 |
| `max_recent_actions` | simulation_runner.py | 50 | 最近动作缓存数量 |

### B. 错误码手册

| 错误码 | HTTP状态 | 说明 | 处理建议 |
|--------|----------|------|----------|
| `E001` | 400 | 文本内容为空 | 检查上传文件 |
| `E002` | 400 | 本体格式错误 | 验证ontology结构 |
| `E003` | 500 | Zep API调用失败 | 检查API密钥，重试 |
| `E004` | 404 | 图谱不存在 | 检查graph_id |
| `E005` | 400 | 模拟状态错误 | 检查当前状态 |
| `E006` | 500 | 模拟进程启动失败 | 检查OASIS环境 |
| `E007` | 500 | 报告生成失败 | 查看agent_log.jsonl |
| `E008` | 408 | 操作超时 | 增加timeout参数 |

### C. 参考资料

- **Zep Cloud文档**: https://docs.getzep.com/
- **OASIS项目**: https://github.com/camel-ai/oasis
- **ReACT论文**: "ReAct: Synergizing Reasoning and Acting in Language Models"
- **GraphRAG论文**: "From Local to Global: A Graph RAG Approach to Query-Focused Summarization"

---

**文档版本**: v1.0.0
**最后更新**: 2025-01-22
**维护者**: MiroFish Team
