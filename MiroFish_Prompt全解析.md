# MiroFish Prompt 全解析

> **理解驱动智能的核心** - 深入解析所有提示词及其应用

---

## 目录

1. [Prompt架构概览](#1-prompt架构概览)
2. [本体生成Prompt](#2-本体生成prompt)
3. [智能配置生成Prompt](#3-智能配置生成prompt)
4. [Agent人设生成Prompt](#4-agent人设生成prompt)
5. [ReACT报告生成Prompt](#5-react报告生成prompt)
6. [Prompt工程最佳实践](#6-prompt工程最佳实践)

---

## 1. Prompt架构概览

### 1.1 Prompt分类体系

```
MiroFish Prompt 体系
├─ 阶段1：图谱构建阶段
│   └─ 本体生成Prompt (ontology_generator.py)
│      └─ 作用：从文本中提取实体类型和关系类型
│
├─ 阶段2：模拟准备阶段
│   ├─ Agent人设生成Prompt (oasis_profile_generator.py)
│   │   ├─ 个人实体prompt (2000字详细人设)
│   │   └─ 机构实体prompt (2000字官方账号设定)
│   │
│   └─ 智能配置生成Prompt (simulation_config_generator.py)
│       ├─ 时间配置prompt (生成作息时间参数)
│       ├─ 事件配置prompt (生成初始事件和热点话题)
│       └─ Agent配置prompt (为每个Agent生成行为参数)
│
└─ 阶段3：报告生成阶段
    └─ ReACT报告Agent Prompt (report_agent.py)
        ├─ 大纲规划prompt (生成报告结构)
        ├─ 章节生成prompt (ReACT循环 + 工具调用)
        └─ 对话prompt (与用户交互)
```

### 1.2 Prompt调用流程

```
用户输入文件
    ↓
【本体生成】
分析文本 → 识别实体类型(10个) + 关系类型(6-10个)
    ↓
GraphRAG构建 → 知识图谱
    ↓
【人设生成】
为每个实体 → LLM生成2000字人设 → Agent Profile
    ↓
【配置生成】
分析需求 + 图谱信息 → 生成时间/事件/Agent配置
    ↓
OASIS模拟 → 动作日志
    ↓
【报告生成】
ReACT循环调用工具 → 分析结果 → 生成报告
```

---

## 2. 本体生成Prompt

### 2.1 核心目标

**输入**：原始文档（如PDF、新闻报道）
**输出**：实体类型 + 关系类型的Schema定义

### 2.2 System Prompt结构

```python
ONTOLOGY_SYSTEM_PROMPT = """你是一个专业的知识图谱本体设计专家。
你的任务是分析给定的文本内容和模拟需求，设计适合**社交媒体舆论模拟**的实体类型和关系类型。

**重要：你必须输出有效的JSON格式数据，不要输出任何其他内容。**

## 核心任务背景

我们正在构建一个**社交媒体舆论模拟系统**。
- 每个实体都是一个可以在社交媒体上发声、互动、传播信息的"账号"或"主体"
- 实体之间会相互影响、转发、评论、回应
- 我们需要模拟舆论事件中各方的反应和信息传播路径

因此，**实体必须是现实中真实存在的、可以在社媒上发声和互动的主体**：

**可以是**：
- 具体的个人（公众人物、当事人、意见领袖、专家学者、普通人）
- 公司、企业（包括其官方账号）
- 组织机构（大学、协会、NGO、工会等）
- 政府部门、监管机构
- 媒体机构（报纸、电视台、自媒体、网站）
- 社交媒体平台本身
- 特定群体代表（如校友会、粉丝团、维权群体等）

**不可以是**：
- 抽象概念（如"舆论"、"情绪"、"趋势"）
- 主题/话题（如"学术诚信"、"教育改革"）
- 观点/态度（如"支持方"、"反对方"）

## 输出格式

请输出JSON格式，包含以下结构：
{
    "entity_types": [...],
    "edge_types": [...],
    "analysis_summary": "对文本内容的简要分析说明（中文）"
}
"""
```

### 2.3 关键设计规则

#### 规则1：实体类型数量和层次

```python
"""
数量要求：必须正好10个实体类型

层次结构要求（必须同时包含具体类型和兜底类型）：

A. 兜底类型（必须包含，放在列表最后2个）：
   - Person: 任何自然人个体的兜底类型
   - Organization: 任何组织机构的兜底类型

B. 具体类型（8个，根据文本内容设计）：
   - Student, Professor, University, Journalist...

为什么需要兜底类型：
- 文本中会出现各种人物，如"中小学教师"、"路人甲"
- 如果没有专门的类型匹配，他们应该被归入 Person
"""
```

#### 规则2：属性命名限制

```python
"""
注意：属性名不能使用系统保留字

保留字：name, uuid, group_id, created_at, summary

推荐使用：full_name, title, role, position, location, description
"""
```

### 2.4 实际应用示例

```python
# 输入：武汉大学宿舍甲醛事件相关文档

# LLM生成的输出：
{
    "entity_types": [
        {
            "name": "Student",
            "description": "University students living in dormitories",
            "attributes": [
                {"name": "major", "description": "Field of study"},
                {"name": "year", "description": "Grade level"}
            ]
        },
        {
            "name": "University",
            "description": "Educational institution",
            "attributes": [
                {"name": "location", "description": "Campus location"}
            ]
        },
        # ... 其他8个类型
        {
            "name": "Person",  # 兜底类型
            "description": "Any individual person not fitting other types"
        },
        {
            "name": "Organization",  # 兜底类型
            "description": "Any organization not fitting other types"
        }
    ],
    "edge_types": [
        {
            "name": "STUDIES_AT",
            "description": "Enrollment relationship",
            "source_targets": [
                {"source": "Student", "target": "University"}
            ]
        },
        {
            "name": "WORKS_FOR",
            "description": "Employment",
            "source_targets": [
                {"source": "Professor", "target": "University"}
            ]
        }
    ],
    "analysis_summary": "文本描述了一个大学宿舍甲醛超标事件，涉及学生、校方、媒体等多方主体..."
}
```

---

## 3. 智能配置生成Prompt

### 3.1 时间配置Prompt

**目标**：生成符合中国人作息习惯的时间模拟参数

```python
prompt = f"""基于以下模拟需求，生成时间模拟配置。

{context_truncated}

## 任务
请生成时间配置JSON。

### 基本原则（仅供参考，需根据具体事件和参与群体灵活调整）：
- 用户群体为中国人，需符合北京时间作息习惯
- 凌晨0-5点几乎无人活动（活跃度系数0.05）
- 早上6-8点逐渐活跃（活跃度系数0.4）
- 工作时间9-18点中等活跃（活跃度系数0.7）
- 晚间19-22点是高峰期（活跃度系数1.5）
- 23点后活跃度下降（活跃度系数0.5）

### 返回JSON格式（不要markdown）

示例：
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "针对该事件的时间配置说明"
}}

字段说明：
- total_simulation_hours: 模拟总时长，24-168小时
- minutes_per_round: 每轮时长，30-120分钟
- agents_per_hour_min/max: 每小时激活Agent数量
- peak_hours: 高峰时段（通常19-22点）
- off_peak_hours: 低谷时段（通常0-5点）
"""
```

**关键点**：
- 根据参与群体特点调整时段
- 学生群体：高峰可能是21-23点
- 媒体：全天活跃
- 官方机构：只在工作时间

### 3.2 事件配置Prompt

**目标**：生成初始事件和热点话题

```python
prompt = f"""基于以下模拟需求，生成事件配置。

模拟需求: {simulation_requirement}

{context_truncated}

## 可用实体类型及示例
{type_info}

## 任务
请生成事件配置JSON：
- 提取热点话题关键词
- 描述舆论发展方向
- 设计初始帖子内容，**每个帖子必须指定 poster_type（发布者类型）**

**重要**: poster_type 必须从上面的"可用实体类型"中选择，这样初始帖子才能分配给合适的 Agent 发布。
例如：官方声明应由 Official/University 类型发布，新闻由 MediaOutlet 发布，学生观点由 Student 发布。

返回JSON格式（不要markdown）：
{{
    "hot_topics": ["关键词1", "关键词2", ...],
    "narrative_direction": "<舆论发展方向描述>",
    "initial_posts": [
        {{"content": "帖子内容", "poster_type": "实体类型（必须从可用类型中选择）"},
        ...
    ],
    "reasoning": "<简要说明>"
}}
"""
```

### 3.3 Agent配置Prompt

**目标**：为每个Agent生成个性化的行为参数

```python
prompt = f"""基于以下信息，为每个实体生成社交媒体活动配置。

模拟需求: {simulation_requirement}

## 实体列表
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## 任务
为每个实体生成活动配置，注意：
- **时间符合中国人作息**：凌晨0-5点几乎不活动，晚间19-22点最活跃
- **官方机构**（University/GovernmentAgency）：
  活跃度低(0.1-0.3)，工作时间(9-17)活动，响应慢(60-240分钟)，影响力高(2.5-3.0)
- **媒体**（MediaOutlet）：
  活跃度中(0.4-0.6)，全天活动(8-23)，响应快(5-30分钟)，影响力高(2.0-2.5)
- **个人**（Student/Person/Alumni）：
  活跃度高(0.6-0.9)，主要晚间活动(18-23)，响应快(1-15分钟)，影响力低(0.8-1.2)

返回JSON格式（不要markdown）：
{{
    "agent_configs": [
        {
            "agent_id": <必须与输入一致>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <发帖频率>,
            "comments_per_hour": <评论频率>,
            "active_hours": [<活跃小时列表，考虑中国人作息>],
            "response_delay_min": <最小响应延迟分钟>,
            "response_delay_max": <最大响应延迟分钟>,
            "sentiment_bias": <-1.0到1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <影响力权重>
        },
        ...
    ]
}}
"""
```

---

## 4. Agent人设生成Prompt

### 4.1 人设生成的关键区分

MiroFish创新性地将实体分为两类，使用不同prompt：

#### 个人实体（Student, Professor, Person等）

```python
def _build_individual_persona_prompt(...) -> str:
    return f"""为实体生成详细的社交媒体用户人设,最大程度还原已有现实情况。

实体名称: {entity_name}
实体类型: {entity_type}
实体摘要: {entity_summary}
实体属性: {attrs_str}

上下文信息:
{context_str}

请生成JSON，包含以下字段：

1. bio: 社交媒体简介，200字
2. persona: 详细人设描述（2000字的纯文本），需包含:
   - 基本信息（年龄、职业、教育背景、所在地）
   - 人物背景（重要经历、与事件的关联、社会关系）
   - 性格特征（MBTI类型、核心性格、情绪表达方式）
   - 社交媒体行为（发帖频率、内容偏好、互动风格、语言特点）
   - 立场观点（对话题的态度、可能被激怒/感动的内容）
   - 独特特征（口头禅、特殊经历、个人爱好）
   - 个人记忆（人设的重要部分，要介绍这个个体与事件的关联）
3. age: 年龄数字（必须是整数）
4. gender: 性别，必须是英文: "male" 或 "female"
5. mbti: MBTI类型（如INTJ、ENFP等）
6. country: 国家（使用中文，如"中国"）
7. profession: 职业
8. interested_topics: 感兴趣话题数组

重要:
- 所有字段值必须是字符串或数字，不要使用换行符
- persona必须是一段连贯的文字描述
- 使用中文（除了gender字段必须用英文male/female）
- 内容要与实体信息保持一致
"""
```

#### 机构实体（University, GovernmentAgency等）

```python
def _build_group_persona_prompt(...) -> str:
    return f"""为机构/群体实体生成详细的社交媒体账号设定,最大程度还原已有现实情况。

实体名称: {entity_name}
实体类型: {entity_type}
实体摘要: {entity_summary}
实体属性: {attrs_str}

上下文信息:
{context_str}

请生成JSON，包含以下字段：

1. bio: 官方账号简介，200字，专业得体
2. persona: 详细账号设定描述（2000字的纯文本），需包含:
   - 机构基本信息（正式名称、机构性质、成立背景、主要职能）
   - 账号定位（账号类型、目标受众、核心功能）
   - 发言风格（语言特点、常用表达、禁忌话题）
   - 发布内容特点（内容类型、发布频率、活跃时间段）
   - 立场态度（对核心话题的官方立场、面对争议的处理方式）
   - 特殊说明（代表的群体画像、运营习惯）
   - 机构记忆（机构在事件中的已有动作与反应）
3. age: 固定填30（机构账号的虚拟年龄）
4. gender: 固定填"other"（机构账号使用other表示非个人）
5. mbti: MBTI类型，用于描述账号风格，如ISTJ代表严谨保守
6. country: 国家（使用中文，如"中国"）
7. profession: 机构职能描述
8. interested_topics: 关注领域数组

重要:
- 所有字段值必须是字符串或数字，不允许null值
- persona必须是一段连贯的文字描述，不要使用换行符
- 使用中文（除了gender字段必须用英文"other"）
- age必须是整数30，gender必须是字符串"other"
- 机构账号发言要符合其身份定位
"""
```

### 4.2 人设示例对比

#### 个人实体人设示例

```json
{
    "bio": "武大计算机专业大三学生，住在湖滨宿舍",
    "persona": "张明是武汉大学计算机专业大三学生，住在湖滨宿舍。性格开朗外向，MBTI为ENFP。关注社会热点，喜欢在社交媒体上分享观点。对宿舍环境问题比较敏感，希望学校能尽快解决。平时喜欢刷微博和抖音，经常参与校园话题讨论...",
    "age": 21,
    "gender": "male",
    "mbti": "ENFP",
    "country": "中国",
    "profession": "学生",
    "interested_topics": ["教育", "社会热点", "科技"]
}
```

#### 机构实体人设示例

```json
{
    "bio": "武汉大学官方账号，发布重要通知和公告",
    "persona": "武汉大学官方微博账号，代表学校立场。发言风格正式、严谨，注重事实准确性。及时发布学校政策回应、重大事件进展。面对争议时会采取官方立场，保持客观中立。注重维护学校声誉，积极回应社会关切...",
    "age": 30,
    "gender": "other",
    "mbti": "ISTJ",
    "country": "中国",
    "profession": "高等教育机构",
    "interested_topics": ["教育政策", "校园动态", "学术成果"]
}
```

---

## 5. ReACT报告生成Prompt

### 5.1 大纲规划Prompt

**目标**：生成报告的目录结构

```python
system_prompt = """你是一个「未来预测报告」的撰写专家，
拥有对模拟世界的「上帝视角」。

【核心理念】
我们构建了一个模拟世界，并向其中注入了特定的「模拟需求」作为变量。
模拟世界的演化结果，就是对未来可能发生情况的预测。

【你的任务】
撰写一份「未来预测报告」，回答：
1. 在我们设定的条件下，未来发生了什么？
2. 各类Agent（人群）是如何反应和行动的？
3. 这个模拟揭示了哪些值得关注的未来趋势和风险？

【报告定位】
- ✅ 这是一份基于模拟的未来预测报告
- ✅ 聚焦于预测结果：事件走向、群体反应、涌现现象、潜在风险
- ✅ 模拟世界中的Agent言行就是对未来人群行为的预测
- ❌ 不是对现实世界现状的分析
- ❌ 不是泛泛而谈的舆情综述

【章节数量限制】
- 最少2个主章节，最多5个主章节
- 每个章节可以有0-2个子章节
- 内容要精炼，聚焦于核心预测发现
- 章节结构由你根据预测结果自主设计

请输出JSON格式的报告大纲，格式如下：
{
    "title": "报告标题",
    "summary": "报告摘要（一句话概括核心预测发现）",
    "sections": [
        {
            "title": "章节标题",
            "description": "章节内容描述",
            "subsections": [
                {"title": "子章节标题", "description": "子章节描述"}
            ]
        }
    ]
}
"""
```

### 5.2 章节生成Prompt（ReACT核心）

**目标**：让LLM通过ReACT循环生成报告内容

```python
system_prompt = f"""你是一个「未来预测报告」的撰写专家，正在撰写报告的一个章节。

报告标题: {outline.title}
报告摘要: {outline.summary}
预测场景（模拟需求）: {self.simulation_requirement}
当前要撰写的章节: {section.title}

═══════════════════════════════════════════════════════════════
【最重要的规则 - 必须遵守】
═══════════════════════════════════════════════════════════════

1. 【必须调用工具观察模拟世界】
   - 你正在以「上帝视角」观察未来的预演
   - 所有内容必须来自模拟世界中发生的事件和Agent言行
   - 禁止使用你自己的知识来编写报告内容
   - 每个章节至少调用2次工具（最多4次）来观察模拟的世界

2. 【必须引用Agent的原始言行】
   - Agent的发言和行为是对未来人群行为的预测
   - 在报告中使用引用格式展示这些预测，例如：
     > "某类人群会表示：原文内容..."
   - 这些引用是模拟预测的核心证据

3. 【忠实呈现预测结果】
   - 报告内容必须反映模拟世界中的代表未来的模拟结果
   - 不要添加模拟中不存在的信息
   - 如果某方面信息不足，如实说明

═══════════════════════════════════════════════════════════════
【⚠️ 格式规范 - 极其重要！】
═══════════════════════════════════════════════════════════════

【一个章节 = 最小内容单位】
- 每个章节是报告的最小分块单位
- ❌ 禁止在章节内使用任何 Markdown 标题（#、##、###、#### 等）
- ❌ 禁止在内容开头添加章节主标题
- ✅ 章节标题由系统自动添加，你只需撰写纯正文内容
- ✅ 使用**粗体**、段落分隔、引用、列表来组织内容

【正确示例】
本章节分析了事件的舆论传播态势。通过对模拟数据的深入分析，我们发现...

**首发引爆阶段**

微博作为舆情的第一现场，承担了信息首发的核心功能：

> "微博贡献了68%的首发声量..."

**情绪放大阶段**

抖音平台进一步放大了事件影响力：
- 视觉冲击力强
- 情绪共鸣度高

【错误示例】
## 执行摘要          ← 错误！不要添加任何标题
### 一、首发阶段     ← 错误！不要用###分小节

### 1.1 详细分析   ← 错误！不要用####细分

本章节分析了...

═══════════════════════════════════════════════════════════════
【可用检索工具】（每章节调用2-4次）
═════════════════════════════════════════════════════════════

- insight_forge: 用于深度分析，会自动分解问题并多维度检索
- panorama_search: 用于了解全貌和演变过程
- quick_search: 用于快速验证某个具体信息
- interview_agents: 用于采访模拟Agent，获取不同角色的真实观点

【ReACT工作流程】
1. Thought: [分析需要什么信息，规划检索策略]
2. Action: [调用工具获取信息]
    <tool_call>
   {{"name": "工具名称", "parameters": {{"参数名": "参数值"}}}}


3. Observation: [分析工具返回结果]
4. 重复步骤1-3，直到收集到足够信息（最多5轮）
5. Final Answer: [基于检索结果撰写章节内容]
"""
```

### 5.3 ReACT工具调用示例

**LLM的学习过程**：

```
Round 1:
  LLM: Thought: 我需要了解舆论传播的整体态势
  LLM: Action: <insight_forge query="舆论传播的主要路径和关键节点">

  Tool: 返回了3条主要传播路径和15个关键节点

Round 2:
  LLM: Thought: 我需要了解不同群体的具体反应
  LLM: Action: <interview_agents interview_topic="学生对甲醛事件的看法">

  Tool: 返回了5个学生的采访记录

Round 3:
  LLM: Thought: 信息已足够，可以开始撰写
  LLM: Final Answer: 本章节分析了...
```

---

## 6. Prompt工程最佳实践

### 6.1 分层设计原则

```
┌─────────────────────────────────────────────────────────────┐
│                  Prompt 分层设计                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Layer 1: 角色定义层                                         │
│  - "你是一个...专家"                                         │
│  - 明确身份和职责                                           │
│  - 设定输出格式                                               │
│                                                             │
│  Layer 2: 任务说明层                                         │
│  - "你的任务是..."                                           │
│  - 核心目标是什么                                           │
│  - 输入是什么                                               │
│                                                             │
│  Layer 3: 约束规则层                                         │
│  - "必须遵守的规则"                                         │
│  - "禁止..."                                               │
│  - 格式要求                                                 │
│                                                             │
│  Layer 4: 示例和参考层                                         │
│  - "示例："...                                                │
│  - "参考原则："...                                           │
│  - 违背约束的后果                                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 避免常见陷阱

#### 陷阱1：幻觉问题

```python
# ❌ 错误示例
prompt = """
你是专家。请生成报告内容。
注意：要在报告中详细描述...
"""

# 问题：LLM可能会编造不存在的内容

# ✅ 正确做法
prompt = """
【最重要的规则】
1. 必须调用工具观察模拟世界
2. 禁止使用你自己的知识
3. 报告内容必须来自模拟结果

如果某方面信息不足，如实说明，不要编造。
"""
```

#### 陷阱2：格式不一致

```python
# ❌ 可能导致解析失败
persona: """
张明是个开朗的学生，
喜欢打篮球。
"""

# ✅ 确保格式正确
persona: "张明是个开朗的学生，喜欢打篮球。"
```

#### 陷阱3：上下文过长

```python
# ❌ 可能超出token限制或导致注意力分散
context = document_text  # 可能有10万字

# ✅ 截断并摘要
context = document_text[:10000] + "\n...(已截断)"
```

### 6.3 可复用的Prompt模式

#### 模式1：分步生成（避免复杂任务一次性失败）

```python
class SimulationConfigGenerator:
    """分步生成配置"""

    def generate_config(self, ...):
        # 步骤1：生成时间配置
        time_config = self._generate_time_config(context)

        # 步骤2：生成事件配置
        event_config = self._generate_event_config(context)

        # 步骤3-N：分批生成Agent配置（每批15个）
        for batch in batches:
            agent_configs = self._generate_agent_configs_batch(batch)
```

#### 模式2：强制约束 + JSON修复

```python
def _call_llm_with_retry(self, prompt: str, system_prompt: str):
    """带重试和修复的LLM调用"""

    for attempt in range(3):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )

            content = response.choices[0].message.content

            # 检查是否被截断
            if response.choices[0].finish_reason == 'length':
                content = self._fix_truncated_json(content)

            # 解析JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                # 尝试修复
                result = self._try_fix_config_json(content)
                if result:
                    return result

        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))

    raise last_error
```

### 6.4 实用的Prompt模板

#### 模板1：中文角色prompt（确保LLM使用中文）

```python
system_prompt = "你是社交媒体用户画像生成专家。
生成详细、真实的人设用于舆论模拟，最大程度还原已有现实情况。
必须返回有效的JSON格式，所有字符串值不能包含未转义的换行符。
使用中文。"
```

#### 模板2：类型约束prompt（确保输出格式正确）

```python
prompt = """
...

返回JSON格式（不要markdown）：
{
    "bio": "简介内容",
    "persona": "详细人设（纯文本，不要换行）",
    "age": 整数,
    "gender": "male 或 female 或 other",
    ...
}

注意：
- gender必须是英文：male/female/other
- 所有字段必须有值，不能为null
- persona必须是连贯的纯文本
"""
```

#### 模板3：验证规则prompt（确保LLM遵守约束）

```python
prompt += """

**必须遵守的规则**：
1. 必须正好输出10个实体类型
2. 最后2个必须是兜底类型：Person（个人兜底）和 Organization（组织兜底）
3. 前8个是根据文本内容设计的具体类型
4. 所有实体类型必须是现实中可以发声的主体，不能是抽象概念
5. 属性名不能使用 name、uuid、group_id 等保留字

验证：在输出前检查是否符合上述规则。
"""
```

---

## 7. Prompt对比矩阵

| 功能模块 | Prompt类型 | 核心约束 | 输出格式 | 关键词 |
|---------|-----------|-----------|---------|--------|
| **本体生成** | 分析型prompt | 10个实体类型，必须有兜底类型 | JSON | "社媒模拟"、"真实存在的主体" |
| **时间配置** | 参数生成prompt | 符合中国人作息 | JSON | "凌晨0-5点几乎无人"、"晚间19-22点最活跃" |
| **事件配置** | 事件设计prompt | poster_type必须匹配实体类型 | JSON | "poster_type 必须从可用类型中选择" |
| **个人人设** | 创作型prompt | 2000字详细人设，包含8个维度 | JSON | "基本信息、人物背景、性格特征、社交媒体行为" |
| **机构人设** | 官方型prompt | 2000字官方设定，机构定位 | JSON | "官方立场、发言风格、工作时间" |
| **Agent配置** | 行为参数prompt | 不同类型有不同活跃度 | JSON | "官方机构：低活跃高影响力"、"个人：高活跃低影响力" |
| **报告大纲** | 规划型prompt | 2-5个章节，结构清晰 | JSON | "最少2个，最多5个主章节" |
| **章节生成** | ReACT循环prompt | 必须调用工具，必须引用原文 | Markdown | "必须调用工具观察模拟世界"、"引用Agent原始言行" |

---

## 8. Prompt调用流程全景

```
┌─────────────────────────────────────────────────────────────┐
│                    MiroFish 完整Prompt调用链                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 用户上传文件 + 需求                                      │
│     ↓                                                      │
│  [本体生成Prompt]                                           │
│     └─ 分析文档 → 生成10种实体类型 + 关系类型                    │
│                                                             │
│  2. 文本分块 + GraphRAG构建                                    │
│     ↓                                                      │
│   [Agent人设生成Prompt - 并行执行]                             │
│     ├─ 对于每个实体（30+个）                                   │
│     │   └─ 个人实体 → 2000字人设（个人记忆）                    │
│     │   └─ 机构实体 → 2000字官方设定（机构记忆）                  │
│     └─ 保存到Twitter/Reddit Profile文件                         │
│                                                             │
│  3. 准备模拟环境                                              │
│     ↓                                                      │
│  [智能配置生成Prompt - 分步执行]                             │
│     ├─ [时间配置Prompt] → 中国人作息时间参数                       │
│     ├─ [事件配置Prompt] → 初始帖子 + 热点话题                   │
│     └─ [Agent配置Prompt] → 行为参数（活跃度、响应速度等）         │
│                                                             │
│   4. OASIS模拟运行                                            │
│     ↓                                                      │
│     Agent行为 → 动作日志 (actions.jsonl)                    │
│                                                             │
│   5. 准备报告生成                                              │
│     ↓                                                      │
│  [大纲规划Prompt]                                          │
│     └─ 分析需求 → 生成报告结构（2-5个章节）                  │
│                                                             │
│  6. 逐章节生成（ReACT循环）                                   │
│     for each chapter:                                          │
│         ├─ [章节生成Prompt]                                   │
│         │   └─ ReACT循环（最多5轮）:                     │
│         │       ├─ Thought → 需要什么信息               │
│         │       ├─ Action → 调用检索工具               │
│         │       ├─ Observation → 分析工具返回           │
│         │       └─ 重复或 Final Answer              │
│         └─ 保存章节内容                                         │
│                                                             │
│  7. 报告完成                                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Prompt效果示例

### 9.1 本体生成Prompt效果

```markdown
### 输入（用户上传的新闻报道）
"武汉大学某宿舍发现甲醛超标..."

### 输出（LLM生成的本体）
```json
{
  "entity_types": [
    {"name": "Student", "description": "武汉大学学生..."},
    {"name": "University", "description": "高等教育机构..."},
    {"name": "GovernmentAgency", "description": "政府部门..."},
    {"name": "MediaOutlet", "description": "新闻媒体..."},
    {"name": "Professor", "description": "高校教师..."},
    {"name": "Alumni", "description": "校友..."},
    {"name": "Parent", "description": "学生家长..."},
    {"name": "Person", "description": "其他个人..."},  # 兜底
    {"name": "Organization", "description": "其他机构..."}  # 兜底
  ],
  "edge_types": [
    {"name": "STUDIES_AT", "source_targets": [{"source": "Student", "target": "University"}]},
    {"name": "WORKS_FOR", "source_targets": [{"source": "Professor", "target": "University"}]},
    ...
  ],
  "analysis_summary": "文本描述了武汉大学宿舍甲醛事件..."
}
```

### 9.2 Agent人设Prompt效果

```markdown
### 输入（实体信息）
实体：张三，Student
摘要：武汉大学计算机专业学生，住在湖滨宿舍

### 输出（LLM生成的2000字人设）
```json
{
  "bio": "武大计算机专业大三学生，住在湖滨宿舍",
  "persona": "张明是武汉大学计算机专业大三学生，住在湖滨宿舍。
性格开朗外向，MBTI为ENFP。关注社会热点，喜欢在社交媒体上分享观点。
对宿舍环境问题比较敏感，希望学校能尽快解决。
平时喜欢刷微博和抖音，经常参与校园话题讨论。
...",
  "age": 21,
  "gender": "male",
  "mbti": "ENFP",
  "country": "中国",
  "profession": "学生",
  "interested_topics": ["教育", "社会热点", "科技"]
}
```

### 9.3 ReACT章节生成Prompt效果

```markdown
### 工具调用过程

**Round 1**:
```
LLM: 我需要了解学生对甲醛事件的整体反应
LLM: Action: <insight_forge query="学生群体对甲醛事件的主要观点和情绪">
Tool: 返回了15个学生的发言记录
```

**Round 2**:
```
LLM: 我需要了解不同角色的态度差异
LLM: Action: <interview_agents interview_topic="校方应对措施是否满意">
Tool: 返回了校方官方的回应和学生评价
```

**Final Answer**:
```
**学生群体反应**

学生群体对甲醛事件表现出强烈的担忧和不满。模拟数据显示：
> "学校到现在还没有给出明确的解决方案"
> "我们宿舍的甲醛检测结果严重超标，但不知道什么时候能解决"

> "感觉学校在拖延时间，不够重视我们的健康"
```
```

---

## 10. Prompt优化技巧

### 10.1 技巧1：Few-Shot学习（给示例）

```python
prompt = """
请生成武汉大学学生的人设。

【示例1】（参考）：
{
    "bio": "武大计算机专业学生",
    "persona": "性格开朗，喜欢...",
    "age": 20
}

【示例2】（参考）：
{
    "bio": "武汉大学官方账号",
    "persona": "代表学校官方立场...",
    "age": 30
}

请根据上述示例模式，为以下实体生成人设：
实体名称：...
实体类型：...
"""
```

### 10.2 技巧2：思维链（让LLM展示推理）

```python
prompt = """
请逐步思考并生成实体类型：

Step 1: 首先列出文档中提到的主要角色（如学生、老师、媒体）
Step 2: 识别这些角色的共同特征和差异
Step 3: 将相似角色归类（如"本科生"、"研究生" → "Student"）
Step 4: 识别缺失的角色类型（添加兜底类型）
Step 5: 最终输出10个实体类型的JSON定义
"""
```

### 10.3 技巧3：格式验证（在prompt中）

```python
prompt += """

【输出前自检】
请检查：
1. 是否正好10个实体类型？_____
2. 最后2个是否为Person和Organization？_____
3. 所有实体类型是否为现实中可发声的主体？_____

确认无误后，再输出JSON。
"""
```

### 10.4 技巧4：约束强化（重复强调）

```python
# 多次重复关键约束
prompt += """

【重要】
1. 输出必须是JSON格式
2. 所有实体必须是现实中可发声的主体
3. 属性名不能用保留字

【再次强调】
- 第9和第10个类型必须是Person和Organization
- 不要使用抽象概念
- 确保类型之间有清晰的区分
"""
```

---

## 11. 总结：Prompt设计的核心原则

| 原则 | 说明 | MiroFish中的实践 |
|------|------|-------------------|
| **角色明确** | 清晰定义LLM的身份和职责 | "你是专家"、"你是预测报告撰写专家" |
| **任务具体** | 明确输入输出，避免歧义 | 输入格式→ 输出格式 |
| **约束明确** | 清晰列出禁止和必须遵守的规则 | "必须调用工具"、"禁止使用自己知识" |
| **示例具体** | 给出参考示例，降低理解成本 | 时间配置、Agent配置示例 |
| **格式严格** | 严格定义输出格式，便于解析 | "JSON格式"、"不要markdown" |
| **上下文截断** | 控制输入长度，避免混乱 | 文档截断、上下文摘要 |
| **验证机制** | 在prompt中或事后验证输出 | 重试机制、JSON修复 |
| **分步执行** | 复杂任务分解为多步 | 配置生成分为4个步骤 |
| **中文优化** | 明确要求使用中文 | "使用中文（除了gender字段）" |

---

**文档版本**: v1.0.0
**最后更新**: 2025-01-22
**维护者**: MiroFish Team

**说明**: 所有Prompt均基于实际代码提取，已去除敏感信息和版权声明
