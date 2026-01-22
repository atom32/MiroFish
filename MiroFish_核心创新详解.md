# MiroFish 核心创新点详解

> **快速复刻指南** - 如果你已经有自研GraphRAG + OASIS，还需要做什么？

---

## 目录

1. [MiroFish的真实技术构成](#1-mirofish的真实技术构成)
2. [必须自己实现的核心模块](#2-必须自己实现的核心模块)
3. [双平台并行模拟架构](#3-双平台并行模拟架构)
4. [智能配置生成系统](#4-智能配置生成系统)
5. [ReACT Report Agent深度解析](#5-react-report-agent深度解析)
6. [Interview通信机制](#6-interview通信机制)
7. [动态图谱记忆更新](#7-动态图谱记忆更新)
8. [最小可行复刻方案](#8-最小可行复刻方案)

---

## 1. MiroFish的真实技术构成

### 1.1 技术依赖分层

```
┌─────────────────────────────────────────────────────────────┐
│                  MiroFish 完整技术栈                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ═══════════════════════════════════════════════════════════│
│  ████ 自研核心 (系统级创新) - 这才是MiroFish的价值           │
│  ═══════════════════════════════════════════════════════════│
│                                                             │
│  Layer 1: 业务编排与状态管理 (50%)                          │
│  ├─ 全流程自动化编排           ⭐⭐⭐⭐⭐                 │
│  ├─ 双平台并行模拟架构           ⭐⭐⭐⭐⭐                 │
│  ├─ 模拟生命周期管理             ⭐⭐⭐⭐⭐                 │
│  ├─ 实时监控与日志系统           ⭐⭐⭐⭐                  │
│  └─ Interview机制              ⭐⭐⭐⭐⭐                 │
│                                                             │
│  Layer 2: 智能决策与推理 (30%)                              │
│  ├─ ReACT Report Agent           ⭐⭐⭐⭐⭐                 │
│  ├─ 智能配置生成                 ⭐⭐⭐⭐                  │
│  ├─ Agent人设生成                ⭐⭐⭐                   │
│  └─ 动态记忆更新                 ⭐⭐⭐⭐                  │
│                                                             │
│  ═══════════════════════════════════════════════════════════│
│  ░░░░ 外部依赖 (可替换) - 这不是MiroFish的创新             ░░░░░
│  ═══════════════════════════════════════════════════════════│
│                                                             │
│  Layer 3: 基础能力 (20%)                                    │
│  ├─ GraphRAG构建      → 你的HippoRAG或其他实现              │
│  ├─ Agent模拟引擎      → OASIS框架                          │
│  └─ LLM推理            → OpenAI/本地模型                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 价值贡献分析

| 模块 | 技术难度 | 创新程度 | MiroFish贡献 |
|------|----------|----------|--------------|
| 双平台并行架构 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 100% 自研 |
| Interview机制 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 100% 自研 |
| ReACT Report Agent | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 80% 自研(工具调用部分) |
| 智能配置生成 | ⭐⭐⭐ | ⭐⭐⭐ | 70% 自研 |
| 状态管理系统 | ⭐⭐⭐ | ⭐⭐ | 100% 自研(工程实现) |
| OASIS集成 | ⭐ | - | 包装调用 |
| GraphRAG | - | - | 外部服务 |

---

## 2. 必须自己实现的核心模块

如果你已经有：
- ✅ 自研GraphRAG (如HippoRAG)
- ✅ OASIS框架

还需要实现这 **7大核心模块** 才能复刻MiroFish：

```
复刻MiroFish = GraphRAG + OASIS + 以下7个模块

┌─────────────────────────────────────────────────────────────┐
│                      必须实现的模块                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 模拟编排引擎        [SimulationOrchestrator]             │
│     └─ 全流程自动化：图谱 → 模拟 → 报告                    │
│                                                             │
│  2. 双平台并行架构      [DualPlatformSimulator]             │
│     └─ Twitter + Reddit 并发运行 + 状态合并                │
│                                                             │
│  3. 智能配置生成器      [SmartConfigGenerator]              │
│     └─ LLM分析需求 → 生成OASIS配置参数                     │
│                                                             │
│  4. 实时监控系统        [RealtimeMonitor]                   │
│     └─ 日志流解析 → 状态更新 → 进度推送                    │
│                                                             │
│  5. Interview通信协议   [InterviewProtocol]                │
│     └─ 运行时与Agent对话的IPC机制                          │
│                                                             │
│  6. ReACT Report Agent   [ReportAgent]                      │
│     └─ ReACT循环 + 工具集成 + 报告生成                     │
│                                                             │
│  7. 动态记忆更新器      [MemoryUpdater]                     │
│     └─ Agent活动实时写回GraphRAG图谱                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 双平台并行模拟架构

### 3.1 为什么这是核心创新？

**问题**：OASIS框架本身只支持**单平台**运行

```python
# OASIS原始用法（单平台）
from oasis.simulation import TwitterSimulation

sim = TwitterSimulation(
    agents=agents,
    config=config
)
sim.run(num_rounds=100)  # 阻塞运行，单平台
```

**MiroFish的创新**：实现了**双平台并行** + **实时监控**

### 3.2 架构设计

```python
class DualPlatformSimulator:
    """
    双平台并行模拟器

    核心创新点：
    1. 同时运行Twitter和Reddit两个OASIS实例
    2. 实时解析两个平台的动作日志
    3. 合并状态到统一的数据结构
    4. 支持运行时Interview交互
    """

    def __init__(self, simulation_id: str):
        self.simulation_id = simulation_id

        # 两个独立的OASIS进程
        self.twitter_proc = None
        self.reddit_proc = None

        # 统一的状态管理
        self.run_state = SimulationRunState(
            simulation_id=simulation_id,
            twitter_running=False,
            reddit_running=False,
            twitter_current_round=0,
            reddit_current_round=0
        )

        # 实时监控线程
        self.monitor_thread = None

    def start_parallel_simulation(self, config_path: str):
        """
        启动双平台并行模拟

        难点：
        1. OASIS是阻塞式运行，如何同时跑两个？
        2. 如何实时获取运行状态？
        3. 两个平台的进度如何同步？
        """

        # 方案：启动两个独立的子进程
        # 每个进程运行一个OASIS实例

        # 启动Twitter进程
        self.twitter_proc = subprocess.Popen([
            sys.executable,
            "run_twitter_simulation.py",
            "--config", config_path,
            "--max-rounds", "144"
        ], cwd=self.sim_dir)

        # 启动Reddit进程
        self.reddit_proc = subprocess.Popen([
            sys.executable,
            "run_reddit_simulation.py",
            "--config", config_path,
            "--max-rounds", "144"
        ], cwd=self.sim_dir)

        # 启动监控线程（非阻塞）
        self.monitor_thread = threading.Thread(
            target=self._monitor_both_platforms,
            daemon=True
        )
        self.monitor_thread.start()

        return self.run_state

    def _monitor_both_platforms(self):
        """
        实时监控双平台运行状态

        核心技术：增量日志解析
        """
        twitter_log = f"{self.sim_dir}/twitter/actions.jsonl"
        reddit_log = f"{self.sim_dir}/reddit/actions.jsonl"

        twitter_pos = 0
        reddit_pos = 0

        while self.twitter_proc.poll() is None or self.reddit_proc.poll() is None:
            # 增量读取Twitter日志
            twitter_pos = self._parse_actions_log(
                twitter_log, twitter_pos, "twitter"
            )

            # 增量读取Reddit日志
            reddit_pos = self._parse_actions_log(
                reddit_log, reddit_pos, "reddit"
            )

            # 检查是否都完成
            if self._check_both_completed():
                self.run_state.status = "completed"
                break

            # 持久化状态
            self._save_state()

            time.sleep(2)  # 每2秒轮询

    def _parse_actions_log(
        self,
        log_path: str,
        position: int,
        platform: str
    ) -> int:
        """
        增量解析动作日志

        技术难点：
        1. 文件可能被其他进程写入，如何避免冲突？
        2. 如何检测新内容？
        3. 如何解析事件类型（round_end, simulation_end）？
        """
        if not os.path.exists(log_path):
            return position

        with open(log_path, 'r') as f:
            f.seek(position)
            for line in f:
                action = json.loads(line)

                # 事件类型检测
                if "event_type" in action:
                    self._handle_event(action, platform)
                else:
                    # 普通动作
                    self._handle_action(action, platform)

        return f.tell()

    def _handle_event(self, event: dict, platform: str):
        """
        处理特殊事件

        关键事件：
        - simulation_start: 模拟开始
        - round_start: 轮次开始
        - round_end: 轮次结束（更新轮数和时间）
        - simulation_end: 模拟结束（标记完成）
        """
        event_type = event["event_type"]

        if event_type == "round_end":
            round_num = event["round"]
            simulated_hours = event["simulated_hours"]

            # 更新平台独立状态
            if platform == "twitter":
                self.run_state.twitter_current_round = round_num
                self.run_state.twitter_simulated_hours = simulated_hours
            else:
                self.run_state.reddit_current_round = round_num
                self.run_state.reddit_simulated_hours = simulated_hours

            # 更新全局状态（取最大值）
            self.run_state.current_round = max(
                self.run_state.twitter_current_round,
                self.run_state.reddit_current_round
            )

        elif event_type == "simulation_end":
            if platform == "twitter":
                self.run_state.twitter_completed = True
                self.run_state.twitter_running = False
            else:
                self.run_state.reddit_completed = True
                self.run_state.reddit_running = False

            # 检查是否都完成
            if self.run_state.twitter_completed and self.run_state.reddit_completed:
                self.run_state.status = "completed"
                self.run_state.completed_at = datetime.now().isoformat()
```

### 3.3 关键技术点

| 技术点 | 难度 | 解决方案 |
|--------|------|----------|
| **进程隔离** | ⭐⭐ | 每个平台独立子进程 + 独立SQLite数据库 |
| **日志格式统一** | ⭐⭐⭐ | 定义统一的actions.jsonl格式 |
| **增量解析** | ⭐⭐⭐ | 维护文件position，seek读取 |
| **状态同步** | ⭐⭐⭐⭐ | 双平台独立计数 + 全局取max |
| **完成检测** | ⭐⭐⭐ | 监听simulation_end事件 |
| **进程清理** | ⭐⭐⭐⭐ | 跨平台信号处理（Windows/Unix） |

### 3.4 OASIS脚本适配

```python
# run_twitter_simulation.py
# 这是MiroFish对OASIS的封装脚本

def main():
    # 1. 读取配置
    config = load_config(args.config)

    # 2. 加载Agent Profile
    profiles = load_twitter_profiles(config)

    # 3. 初始化OASIS Twitter环境
    from oasis.simulation import TwitterSimulationEnv
    env = TwitterSimulationEnv(
        agent_profiles=profiles,
        **config['twitter_config']
    )

    # 4. 运行模拟
    env.run(
        num_rounds=config['total_rounds'],
        action_logger=JSONLogger("twitter/actions.jsonl")  # 关键：日志输出
    )

    # 5. 写入结束事件
    with open("twitter/actions.jsonl", "a") as f:
        f.write(json.dumps({
            "event_type": "simulation_end",
            "total_rounds": config['total_rounds'],
            "total_actions": env.get_total_actions()
        }))

if __name__ == "__main__":
    main()
```

**关键点**：OASIS本身没有这些，需要MiroFish自己实现：
- ✅ JSONLogger（自定义日志格式）
- ✅ 统一的action schema
- ✅ event_type机制（round_start, round_end, simulation_end）

---

## 4. 智能配置生成系统

### 4.1 为什么需要这个？

**问题**：OASIS需要大量配置参数

```python
# OASIS需要的配置（20+ 个参数）
oasis_config = {
    # 时间配置
    "total_simulation_hours": 72,      # 模拟多长时间？
    "minutes_per_round": 30,           # 每轮多少分钟？
    "start_time": "2025-01-22T08:00",  # 什么时候开始？

    # Agent活跃度
    "post_frequency": 0.3,             # 发帖概率
    "comment_frequency": 0.5,          # 评论概率
    "like_frequency": 0.7,             # 点赞概率
    "follow_frequency": 0.1,           # 关注概率

    # 平台特性
    "twitter_char_limit": 280,         # Twitter字数限制
    "reddit_subreddit_rules": [...],   # Reddit版规

    # ... 还有10+个参数
}
```

**用户只知道**："我想预测未来3天的舆情走向"

**需要系统自动**：分析需求 → 生成合理配置

### 4.2 智能配置生成器设计

```python
class SimulationConfigGenerator:
    """
    智能配置生成器

    核心创新：
    用LLM理解用户的模糊需求，生成具体的OASIS配置参数
    """

    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,  # "预测未来3天的舆情走向"
        document_text: str,
        entities: List[Dict],
        enable_twitter: bool,
        enable_reddit: bool
    ) -> SimulationParameters:
        """
        智能生成模拟配置

        流程：
        ┌─────────────────────────────────────────────────────────┐
        │ 1. 获取上下文信息                                       │
        │    - 图谱统计（实体数量、关系数量）                      │
        │    - 文档摘要                                           │
        │    - 实体类型分布                                       │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ 2. 构建LLM提示                                          │
        │    System Prompt: 定义配置参数的含义和范围              │
        │    User Prompt: 描述需求 + 提供上下文                   │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ 3. 调用LLM生成配置                                      │
        │    - 分析时间需求（"未来3天" → 72小时）                  │
        │    - 分析场景类型（舆情 → 高活跃度）                    │
        │    - 生成所有20+个参数的值                             │
        │    - 生成推理说明（为什么这样配置）                     │
        └─────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────────────────┐
        │ 4. 验证与后处理                                        │
        │    - 检查参数合理性（总轮数 = 总时长 / 每轮时长）        │
        │    - 边界检查（概率在0-1之间）                          │
        │    - 类型转换（字符串 → 整数/浮点数）                   │
        └─────────────────────────────────────────────────────────┘
        """

        # 步骤1：获取上下文
        context = self._gather_context(graph_id, entities)

        # 步骤2：构建提示
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(
            simulation_requirement, document_text, context
        )

        # 步骤3：LLM生成
        response = self.llm.chat_json([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        # 步骤4：解析并验证
        params = self._parse_and_validate(response)

        return SimulationParameters(
            simulation_id=simulation_id,
            **params,
            generation_reasoning=response.get("reasoning", "")
        )

    def _build_system_prompt(self) -> str:
        """
        System Prompt: 定义配置参数的含义

        关键点：要给出具体的取值范围和默认值
        """
        return """你是一个OASIS多智能体模拟系统的配置专家。

你的任务是：根据用户的预测需求，生成合理的模拟配置参数。

【配置参数说明】

1. 时间配置：
   - total_simulation_hours: 总模拟时长（小时）
     * 舆情预测：通常24-168小时（1-7天）
     * 金融预测：通常24-720小时（1-30天）
     * 创意写作：通常灵活设置

   - minutes_per_round: 每轮模拟时长（分钟）
     * 通常30-120分钟
     * 越小越精细，但模拟轮次越多

   - start_time: 模拟开始时间
     * 通常为当前时间或指定时间

2. Agent活跃度配置（0-1之间的概率值）：
   - post_frequency: 发帖概率
     * 舆情热点事件：0.4-0.7（高活跃）
     * 日常讨论：0.1-0.3（低活跃）

   - comment_frequency: 评论概率
     * 通常高于发帖概率（0.5-0.8）

   - like_frequency: 点赞概率
     * 通常最高（0.6-0.9）

   - follow_frequency: 关注概率
     * 通常最低（0.05-0.2）

3. 平台特性配置：
   - twitter_char_limit: Twitter字数限制（默认280）
   - reddit_subreddit_rules: Reddit版规（JSON格式）

【输出格式】
请输出JSON格式的配置，包含所有参数及其解释：
{
    "time_config": {
        "total_simulation_hours": 数值,
        "minutes_per_round": 数值,
        "start_time": "ISO时间字符串"
    },
    "agent_activity": {
        "post_frequency": 0-1的小数,
        "comment_frequency": 0-1的小数,
        "like_frequency": 0-1的小数,
        "follow_frequency": 0-1的小数
    },
    "reasoning": "配置推理说明：为什么这样设置"
}
"""

    def _build_user_prompt(
        self,
        requirement: str,
        document_text: str,
        context: dict
    ) -> str:
        """
        User Prompt: 描述具体需求
        """
        return f"""【用户的预测需求】
{requirement}

【背景材料】
文档摘要：{context['document_summary']}
实体数量：{context['entity_count']}
实体类型：{context['entity_types']}

【任务】
根据上述需求，生成最合适的OASIS模拟配置参数。

【重要】
1. 理解需求中的时间关键词（如"未来3天" → 72小时）
2. 根据场景类型调整活跃度（舆情→高活跃，日常→低活跃）
3. 确保配置参数之间的一致性（总轮数 = 总时长 / 每轮时长）
4. 提供清晰的推理说明
"""
```

### 4.3 实际效果示例

```python
# 输入
requirement = "预测未来3天，武汉大学宿舍甲醛事件在学生群体中的舆情走向"

# LLM生成的配置
{
    "time_config": {
        "total_simulation_hours": 72,  # 3天 = 72小时
        "minutes_per_round": 30,       # 30分钟一轮，足够精细
        "start_time": "2025-01-22T08:00:00"
    },
    "agent_activity": {
        "post_frequency": 0.6,        # 舆情事件，高活跃
        "comment_frequency": 0.7,     # 评论多于发帖
        "like_frequency": 0.8,        # 点赞最频繁
        "follow_frequency": 0.15      # 关注行为较少
    },
    "reasoning": "根据'未来3天'的需求，设置总模拟时长为72小时。考虑到'宿舍甲醛事件'是热点话题，学生群体会高度关注，因此设置了较高的活跃度参数（post_frequency=0.6）。每轮30分钟的设置能够捕捉到舆情的快速演变。"
}
```

### 4.4 关键技术点

| 技术点 | 难度 | 解决方案 |
|--------|------|----------|
| **需求理解** | ⭐⭐⭐ | LLM自然语言理解 |
| **参数映射** | ⭐⭐⭐⭐ | 提示工程 + 上下文注入 |
| **合理性验证** | ⭐⭐⭐ | 后处理规则检查 |
| **可解释性** | ⭐⭐ | 要求LLM输出推理过程 |

---

## 5. ReACT Report Agent深度解析

### 5.1 ReACT框架简介

**ReACT = Reasoning + Acting**

```
传统LLM：
Question → LLM → Answer

ReACT：
Question → LLM → Thought → Action → Observation → LLM → Thought → Action → ... → Answer
```

### 5.2 MiroFish的ReACT实现

```python
class ReportAgent:
    """
    ReACT Report Agent

    核心创新：将ReACT与多工具深度集成，生成高质量的预测报告
    """

    def _generate_section_react(
        self,
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str]
    ) -> str:
        """
        使用ReACT模式生成单个章节

        ReACT循环：
        ┌─────────────────────────────────────────────────────────┐
        │                                                         │
        │  while iteration < MAX_ITERATIONS:                      │
        │      1. Thought (LLM)                                   │
        │         → 我需要什么信息？                               │
        │                                                         │
        │      2. Action (解析并执行工具调用)                     │
        │         → 调用 insight_forge / panorama_search 等        │
        │                                                         │
        │      3. Observation (工具返回结果)                      │
        │         → 分析检索到的数据                               │
        │                                                         │
        │      4. 判断                                           │
        │         → 信息足够？ → Final Answer                     │
        │         → 不够？ → 继续下一轮                            │
        │                                                         │
        └─────────────────────────────────────────────────────────┘
        """

        # 构建系统提示（关键！）
        system_prompt = f"""你是一个「未来预测报告」的撰写专家。

报告标题: {outline.title}
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

【可用工具】
- insight_forge: 深度洞察检索（最强大，自动分解问题）
- panorama_search: 广度搜索（获取全貌，含历史内容）
- quick_search: 简单搜索（快速验证）
- interview_agents: 真实Agent采访（获取不同角色观点）

【格式规范】
- 禁止使用任何标题（#、##等），用**粗体**代替
- 使用 > 格式引用原文
"""

        # 构建用户提示
        user_prompt = f"""【已完成的章节】
{self._format_previous_sections(previous_sections)}

【当前任务】
撰写章节: {section.title}

请开始：
1. 首先思考（Thought）这个章节需要什么信息
2. 然后调用工具（Action）获取模拟数据
3. 收集足够信息后输出 Final Answer
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # ReACT循环
        tool_calls_count = 0

        for iteration in range(5):  # 最多5轮
            # 调用LLM
            response = self.llm.chat(messages, temperature=0.5)

            # 检查Final Answer
            if "Final Answer:" in response:
                if tool_calls_count < 2:
                    # 强制继续调用工具
                    messages.append({
                        "role": "user",
                        "content": "你只调用了1次工具，请至少调用2次获取更多信息"
                    })
                    continue

                # 提取最终答案
                final_answer = response.split("Final Answer:")[-1].strip()
                return final_answer

            # 解析工具调用
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # 提示需要调用工具
                messages.append({
                    "role": "user",
                    "content": "请调用工具获取模拟数据，不要使用自己的知识"
                })
                continue

            # 执行工具调用
            tool_results = []
            for call in tool_calls:
                result = self._execute_tool(
                    call["name"],
                    call["parameters"]
                )
                tool_results.append(f"工具 {call['name']} 返回:\n{result}")
                tool_calls_count += 1

            # 添加到上下文
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation:\n" + "\n\n".join(tool_results)
            })

        # 强制返回
        return response
```

### 5.3 工具集成示例

```python
def _execute_tool(self, tool_name: str, parameters: dict) -> str:
    """
    执行工具调用

    工具类型：
    1. 图谱检索工具（调用你的GraphRAG）
    2. Agent采访工具（通过IPC调用OASIS）
    """

    if tool_name == "insight_forge":
        # 深度洞察检索
        query = parameters["query"]

        # 调用你的HippoRAG
        result = your_hipporage.deep_search(
            graph_id=self.graph_id,
            query=query,
            strategy="traversal"  # HippoRAG特色：图遍历
        )

        return f"找到{len(result['entities'])}个相关实体，{len(result['relations'])}条关系"

    elif tool_name == "interview_agents":
        # Agent采访
        interview_topic = parameters["interview_topic"]

        # 通过IPC调用OASIS
        result = self.interview_agent(
            agent_id=5,  # 选择一个相关Agent
            prompt=interview_topic
        )

        return f"Agent回答：{result['response']}"
```

### 5.4 关键技术点

| 技术点 | 难度 | 解决方案 |
|--------|------|----------|
| **提示工程** | ⭐⭐⭐⭐ | 详细的System Prompt + 示例 |
| **工具解析** | ⭐⭐⭐ | 正则表达式匹配XML/JSON格式 |
| **工具执行** | ⭐⭐⭐ | 统一的_execute_tool接口 |
| **循环控制** | ⭐⭐⭐ | 最少工具调用次数检查 |
| **上下文管理** | ⭐⭐⭐⭐ | 限制历史长度，避免token爆炸 |

---

## 6. Interview通信机制

### 6.1 为什么需要Interview？

**问题**：OASIS运行在独立的子进程中，如何与运行中的Agent对话？

```python
# 场景：模拟运行到第50轮，用户想采访某个Agent
# OASIS进程正在运行，如何向它发送命令？

# 需要一个跨进程通信机制（IPC）
```

### 6.2 IPC方案设计

```python
class SimulationIPCClient:
    """
    模拟进程通信客户端

    核心创新：实现运行时与OASIS Agent的实时对话

    技术方案：
    - Windows: 命名管道 (Named Pipes)
    - Unix: Unix Domain Sockets
    """

    def __init__(self, sim_dir: str):
        self.sim_dir = sim_dir

        # IPC通道
        if sys.platform == 'win32':
            self.pipe_path = f"\\\\.\\pipe\\mirofish_{self.sim_id}"
        else:
            self.pipe_path = f"/tmp/mirofish_{self.sim_id}.sock"

    def send_interview(
        self,
        agent_id: int,
        prompt: str,
        platform: str = None,
        timeout: float = 60.0
    ) -> IPCResponse:
        """
        发送Interview命令

        流程：
        ┌─────────────────────────────────────────────────────────┐
        │ 1. 连接到IPC通道                                       │
        │                                                         │
        │ 2. 发送命令                                            │
        │    {                                                   │
        │      "command_type": "interview",                      │
        │      "agent_id": 5,                                    │
        │      "prompt": "你对最近的事件有什么看法？"              │
        │      "platform": null  # 双平台都采访                   │
        │    }                                                   │
        │                                                         │
        │ 3. 等待响应（带超时）                                  │
        │                                                         │
        │ 4. 接收结果                                            │
        │    {                                                   │
        │      "status": "completed",                            │
        │      "result": {                                       │
        │        "twitter_response": "...",                      │
        │        "reddit_response": "..."                        │
        │      }                                                 │
        │    }                                                   │
        └─────────────────────────────────────────────────────────┘
        """

        # 1. 连接
        pipe = self._connect()

        # 2. 发送命令
        command = {
            "command_type": "interview",
            "agent_id": agent_id,
            "prompt": prompt,
            "platform": platform,
            "timestamp": datetime.now().isoformat()
        }
        pipe.send_json(command)

        # 3. 等待响应
        response = pipe.recv_json(timeout=timeout)

        return IPCResponse(**response)

    def check_env_alive(self) -> bool:
        """检查模拟环境是否存活"""
        try:
            pipe = self._connect(timeout=1)
            pipe.send_json({"command_type": "ping"})
            response = pipe.recv_json(timeout=1)
            return response.get("status") == "pong"
        except:
            return False
```

### 6.3 OASIS侧的IPC服务器

```python
# 在OASIS脚本中实现IPC服务器
def main():
    # 初始化OASIS环境
    env = TwitterSimulationEnv(...)

    # 启动IPC服务器（后台线程）
    ipc_server = IPCServer(
        pipe_path=f"\\\\.\\pipe\\mirofish_{sim_id}",
        env=env
    )
    ipc_server.start()

    # 运行模拟（主线程）
    env.run(num_rounds=144)


class IPCServer(threading.Thread):
    """
    IPC服务器（运行在OASIS进程中）

    功能：接收外部命令，调用OASIS环境的方法，返回结果
    """

    def __init__(self, pipe_path: str, env: TwitterSimulationEnv):
        self.pipe_path = pipe_path
        self.env = env
        self.running = True

    def run(self):
        """服务器主循环"""
        # 创建命名管道/Unix Socket
        server = self._create_server(self.pipe_path)

        while self.running:
            # 接受连接
            conn = server.accept()

            try:
                # 接收命令
                command = conn.recv_json()

                # 处理命令
                if command["command_type"] == "interview":
                    result = self._handle_interview(command)
                    conn.send_json(result)

                elif command["command_type"] == "ping":
                    conn.send_json({"status": "pong"})

                elif command["command_type"] == "close_env":
                    self.running = False
                    conn.send_json({"status": "closed"})

            except Exception as e:
                conn.send_json({
                    "status": "error",
                    "error": str(e)
                })

    def _handle_interview(self, command: dict) -> dict:
        """处理Interview命令"""
        agent_id = command["agent_id"]
        prompt = command["prompt"]

        # 调用OASIS的interview方法
        response = self.env.interview_agent(
            agent_id=agent_id,
            prompt=prompt
        )

        return {
            "status": "completed",
            "result": {
                "response": response
            },
            "timestamp": datetime.now().isoformat()
        }
```

### 6.4 关键技术点

| 技术点 | 难度 | 解决方案 |
|--------|------|----------|
| **跨平台IPC** | ⭐⭐⭐⭐ | Windows命名管道 / Unix Socket |
| **命令协议设计** | ⭐⭐⭐ | 统一的JSON格式命令 |
| **异步处理** | ⭐⭐⭐⭐ | OASIS运行时如何响应IPC？ |
| **超时控制** | ⭐⭐⭐ | poll + select机制 |
| **进程清理** | ⭐⭐⭐ | 优雅关闭IPC连接 |

---

## 7. 动态图谱记忆更新

### 7.1 为什么需要？

**问题**：Agent在模拟中产生了新的记忆，如何更新到GraphRAG图谱？

```python
# 场景：
# Round 1: Agent A 说 "甲醛检测超标很担心"
# Round 10: Agent A 改变观点 "检测结果正常，放心了"
# Round 50: Agent B 说 "我记得A之前很担心"

# 需要将Agent的活动和观点变化动态写入图谱
```

### 7.2 动态更新机制

```python
class ZepGraphMemoryUpdater(threading.Thread):
    """
    图谱记忆更新器

    核心创新：
    实时将Agent的活动写回GraphRAG图谱，实现记忆的持久化
    """

    def __init__(self, simulation_id: str, graph_id: str):
        self.simulation_id = simulation_id
        self.graph_id = graph_id
        self.updater_queue = queue.Queue()
        self.running = True

    def run(self):
        """更新器主循环"""
        while self.running:
            try:
                # 从队列获取活动
                activity = self.updater_queue.get(timeout=1)

                # 写入图谱
                self._update_graph_memory(activity)

            except queue.Empty:
                continue

    def add_activity(self, action: dict, platform: str):
        """
        添加Agent活动到更新队列

        从监控线程调用：
        """
        self.updater_queue.put({
            "action": action,
            "platform": platform
        })

    def _update_graph_memory(self, activity: dict):
        """
        将Agent活动写入图谱

        策略：
        1. 提取关键信息（实体、关系、观点）
        2. 创建新的episode
        3. 关联到对应实体
        """
        action = activity["action"]
        platform = activity["platform"]

        # 构建episode内容
        episode_content = f"""
        在{platform}平台上，{action['agent_name']}进行了以下活动：
        - 动作类型：{action['action_type']}
        - 动作参数：{action.get('action_args', {})}
        - 发生时间：{action['timestamp']}
        - 所在轮次：{action['round_num']}
        """

        # 调用你的GraphRAG API（如HippoRAG）
        # 创建新的记忆节点/边
        your_graph_api.add_episode(
            graph_id=self.graph_id,
            entity_name=action['agent_name'],
            content=episode_content,
            timestamp=action['timestamp'],
            metadata={
                "platform": platform,
                "action_type": action['action_type'],
                "round": action['round_num']
            }
        )
```

### 7.3 使用场景

```python
# 在监控线程中启用
def _monitor_simulation(cls, simulation_id: str):
    # 创建更新器
    if enable_graph_memory_update:
        updater = ZepGraphMemoryUpdater(simulation_id, graph_id)
        updater.start()

    # 监控循环
    while process.poll() is None:
        actions = cls._read_action_log(...)

        for action in actions:
            # 添加到更新队列
            updater.add_activity(action, platform)
```

### 7.4 关键技术点

| 技术点 | 难度 | 解决方案 |
|--------|------|----------|
| **实时更新** | ⭐⭐⭐ | 异步队列 + 后台线程 |
| **性能影响** | ⭐⭐⭐⭐ | 批量写入，降低API调用 |
| **数据一致性** | ⭐⭐⭐⭐ | 事务性更新 |
| **图谱演化** | ⭐⭐⭐⭐⭐ | 时效性边（valid_at/invalid_at） |

---

## 8. 最小可行复刻方案

### 8.1 复刻路线图

如果你已经有：
- ✅ 自研GraphRAG (如HippoRAG)
- ✅ OASIS框架

**最少需要实现这4个模块**才能跑起来：

```
MVP（最小可行产品）= GraphRAG + OASIS + 4个模块

┌─────────────────────────────────────────────────────────────┐
│  必须实现的4个核心模块（按优先级）                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  P0 (必须有，否则系统跑不起来)                               │
│  ├─ [1] 模拟编排引擎         2-3周                          │
│  │    └─ 全流程自动化：GraphRAG → OASIS → 报告              │
│  │                                                            │
│  ├─ [2] 智能配置生成器       1-2周                          │
│  │    └─ LLM生成OASIS配置参数                               │
│  │                                                            │
│  P1 (核心功能，影响用户体验)                                 │
│  ├─ [3] 实时监控系统          1-2周                          │
│  │    └─ 日志解析 + 状态更新                                │
│  │                                                            │
│  └─ [4] ReACT Report Agent    2-3周                          │
│       └─ 报告生成 + 工具集成                                  │
│                                                             │
│  P2 (锦上添花，可以后续加)                                   │
│  ├─ Interview机制         1-2周                             │
│  ├─ 双平台并行架构         2-3周                             │
│  └─ 动态记忆更新器         1周                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘

总计：P0 = 3-5周，P0+P1 = 5-8周，全部 = 9-13周
```

### 8.2 快速开始：Week 1-2

```python
# 1. 模拟编排引擎骨架
class SimulationOrchestrator:
    """
    全流程自动化编排

    输入：用户上传的文件 + 预测需求
    输出：预测报告
    """

    def run_full_pipeline(
        self,
        file_path: str,
        simulation_requirement: str
    ) -> dict:
        """
        完整流程

        步骤：
        1. 文件解析 → 文本
        2. 你的GraphRAG → 图谱
        3. 实体抽取 → Agent列表
        4. 智能配置生成 → OASIS配置
        5. OASIS运行 → 模拟数据
        6. ReACT Report Agent → 报告
        """

        # 步骤1：文件解析
        text = self.parse_file(file_path)

        # 步骤2：调用你的GraphRAG
        graph_id = your_graph_rag.build(text)

        # 步骤3：实体抽取
        entities = your_graph_rag.get_entities(graph_id)

        # 步骤4：生成OASIS配置
        config = self.generate_oasis_config(
            simulation_requirement,
            entities
        )

        # 步骤5：运行OASIS
        sim_result = self.run_oasis_simulation(config)

        # 步骤6：生成报告
        report = self.generate_report(
            graph_id=graph_id,
            sim_result=sim_result,
            requirement=simulation_requirement
        )

        return report
```

### 8.3 实现优先级建议

| 优先级 | 模块 | 功能 | 工作量 | 替代方案 |
|--------|------|------|--------|----------|
| **P0-1** | 模拟编排 | 流程串联 | 2周 | 手动执行各步骤 |
| **P0-2** | 智能配置 | 参数生成 | 1周 | 硬编码默认参数 |
| **P1-3** | 实时监控 | 进度追踪 | 1周 | 等待完成后看结果 |
| **P1-4** | ReACT报告 | 自动报告 | 2周 | 手动分析数据 |
| **P2-5** | Interview | 运行时交互 | 1周 | 读取日志分析 |
| **P2-6** | 双平台 | 并行模拟 | 2周 | 单平台依次运行 |
| **P2-7** | 动态记忆 | 图谱更新 | 1周 | 模拟后批量更新 |

### 8.4 技术栈清单

```
复刻MiroFish需要的技术栈：

基础设施：
├─ Python 3.11+
├─ Flask/FastAPI (后端API)
├─ Vue.js 3 (前端界面)
└─ SQLite (OASIS数据存储)

核心依赖：
├─ 你的GraphRAG (如HippoRAG)
├─ OASIS框架
├─ OpenAI SDK (或其他LLM)
└─ 任务队列 (Celery/内存队列)

可选组件：
├─ Redis (缓存、消息队列)
├─ WebSocket (实时推送)
└─ Prometheus (监控)
```

---

## 9. 总结：MiroFish的核心价值

### 9.1 技术贡献矩阵

```
┌─────────────────────────────────────────────────────────────┐
│                  MiroFish的核心创新                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  系统级创新 (80%) - 这是你复刻需要重点关注的               │
│  ├─ 双平台并行模拟架构     ⭐⭐⭐⭐⭐                     │
│  ├─ 全流程自动化编排       ⭐⭐⭐⭐⭐                     │
│  ├─ Interview机制          ⭐⭐⭐⭐⭐                     │
│  ├─ 实时监控系统           ⭐⭐⭐⭐                       │
│  ├─ 智能配置生成           ⭐⭐⭐⭐                       │
│  └─ ReACT报告生成          ⭐⭐⭐⭐                       │
│                                                             │
│  工程实现 (20%)                                           │
│  └─ 状态管理、API设计、前端交互等                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

不是MiroFish创新的（可以直接用现成的）：
├─ GraphRAG算法 → 用你的HippoRAG
├─ Agent模拟引擎 → 用OASIS框架
├─ LLM推理 → 用OpenAI/本地模型
└─ ReACT框架 → 公开的研究成果
```

### 9.2 快速复刻检查清单

```
如果我要复刻MiroFish，需要：

Week 1-2: 基础框架
□ 文件解析模块
□ GraphRAG集成（你的HippoRAG）
□ 基础API框架

Week 3-4: 核心流程
□ 模拟编排引擎（流程串联）
□ 智能配置生成器（LLM生成参数）
□ OASIS集成（单平台）

Week 5-6: 监控与报告
□ 实时监控系统（日志解析）
□ ReACT Report Agent（基础版）

Week 7+: 高级功能
□ Interview机制
□ 双平台并行
□ 动态记忆更新
```

### 9.3 最后的建议

**不要重新发明轮子**，要站在巨人肩膀上：

```
✅ 应该复用的：
- OASIS框架（已成熟的Agent模拟）
- ReACT思想（已验证的推理模式）
- 你的GraphRAG（已验证的算法）

✅ 应该自己实现的：
- 业务流程编排（你的核心创新）
- 双平台并行架构（MiroFish的特色）
- Interview机制（运行时交互）
- 完整的端到端自动化（用户体验）

✅ 可以简化的：
- 从单平台开始，再扩展到双平台
- 从固定配置开始，再加智能生成
- 从离线报告开始，再加实时交互
```

---

**文档版本**: v2.0.0 (核心创新详解版)
**最后更新**: 2025-01-22
**补充说明**: 本文档聚焦于MiroFish的系统级创新，不包含GraphRAG和OASIS等外部依赖的细节
