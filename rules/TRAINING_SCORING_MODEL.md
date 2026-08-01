# 拳击训练视频评分模型 v1.0

> 版本：v1.0（2026年5月）| 训练视频专用模型 · 与实战对练模型 v3.0 并行  
> 适用场景：打手靶（Pad Work）/ 打沙袋（Heavy Bag）/ 空击（Shadow Boxing）  
> 核心差异：评估训练质量，非比赛能力

---

## 1. 模型概述

### 1.1 为什么需要独立模型？

实战对练模型（v3.0）评估的是**比赛能力**（攻防转换、战术执行、对抗适应）。  
训练视频评估的是**训练质量**（技术执行、力量输出、节奏控制）。

**同一拳手，两种视频类型可能得分差异很大**：
- 实战高分 → 比赛能力强，但不代表训练质量高（可能划水）
- 训练高分 → 训练态度好、技术执行标准，但不代表能打赢（缺少对抗压力测试）

**结论**：两套模型独立评分，分别用于不同目的。

---

### 1.2 模型适用范围

| 视频类型 | 代码标识 | 典型时长 | 评估重点 |
|-----------|-----------|-----------|----------|
| 打手靶 | `pad_work` | 3-5分钟 | 技术执行 + 教练互动 |
| 打沙袋 | `heavy_bag` | 3-5分钟 | 力量输出 + 耐力保持 |
| 空击 | `shadow_boxing` | 2-3分钟 | 技术执行 + 节奏感 |

**不适用**：实战对练（使用 v3.0 实战模型）

---

### 1.3 与实战模型的核心差异

| 维度 | 实战模型 v3.0 | 训练模型 v1.0 |
|-------|----------------|---------------|
| 评估目标 | 比赛能力 | 训练质量 |
| 对抗压力 | ✅ 有（对手反击） | ❌ 无（固定靶/无对手） |
| 维度数量 | 14维 | 5维（简化） |
| 数据来源 | 实战视频CV分析 | 训练视频CV分析 |
| 评分输出 | 0-100分（比赛能力） | 0-100分（训练质量） |
| 排名用途 | P4P实战榜 | P4P训练榜 |

---

### 1.4 视频质量要求

#### 最小视频长度

**要求**：≥ 1分钟（60秒）

**理由**：
- < 1分钟：样本量不足（出拳数<20），无法可靠评估"训练强度"（后段掉速）
- 建议长度：2-5分钟（标准训练回合）

**处理方式**：
- 视频< 1分钟：拒绝评分，提示"视频过短，请提供≥1分钟的训练视频"
- 视频1-2分钟：评分但标注"⚠️ 视频较短，评分可靠性中等"

#### 自动检测休息时段

**定义**：静止 > 5秒 = 休息时段（拳手停止出拳，可能喝水/听教练指导）

**检测逻辑**：
```python
REST_THRESHOLD = 5  # 秒
non_punch_gaps = find_gaps_without_punches(video_timestamps)
rest_periods = [gap for gap in non_punch_gaps if gap > REST_THRESHOLD]
```

**评估时处理**：
- 休息时段**不计入**"训练强度"评估（跳过）
- 休息时段**不计入**"节奏控制"评估（跳过）
- 总训练时长 = 有效训练时长（不含休息）

**示例**：
```
视频总时长：180秒（3分钟）
休息时段：0:45-0:52（7秒）、1:30-1:38（8秒）、2:45-2:55（10秒）
有效训练时长：180 - 25 = 155秒
评估基于：155秒有效训练
```

---

## 2. 核心五维框架

### 2.1 维度总览

```
训练质量总分 = 
  技术执行 × 0.30 +
  力量表现 × 0.25 +
  节奏控制 × 0.20 +
  训练强度 × 0.15 +
  技术多样性 × 0.10
```

**设计原则**：
- 所有维度必须可CV自动测量（不依赖教练主观评价）
- 维度定义与实战模型不重叠（避免双重计数）
- 权重可根据训练类型调整（见第3章）

---

### 2.2 维度1：技术执行（30%）

**定义**：出拳技术质量（轨迹标准性、连贯性、还原速度、护手保持）

#### CV测量指标

| 指标 | 权重 | 测量方法 | MVP可行性 |
|------|--------|----------|-------------|
| 出拳轨迹标准性 | 35% | 光流分析：拳头像素位移方向 vs 标准方向向量 | ⚠️ 需开发 |
| 出拳还原速度 | 30% | 连续帧：拳头从目标点→护手位置帧数 | ✅ 可实现 |
| 组合拳连贯性 | 20% | 出拳时间戳差值（组合内间隔） | ✅ 可实现 |
| 护手保持度 | 15% | 非出拳时刻，手部Y坐标 < 肩部Y坐标 | ✅ 可实现（简化版） |

#### 计算公式

```python
# 1. 出拳轨迹标准性 (0-1)
trajectory_scores = []
for punch in punches:
    direction_vector = calculate_direction(punch.start_pos, punch.end_pos)
    standard_vector = STANDARD_DIRECTIONS[punch.type]  # jab: (1,0), hook: (0,1)
    similarity = cosine_similarity(direction_vector, standard_vector)
    trajectory_scores.append(similarity)
trajectory_score = mean(trajectory_scores)

# 2. 出拳还原速度 (0-1)
recovery_times = [punch.recovery_frames for punch in punches]
recovery_score = mean([max(0, 1 - (t - 5) / 20) for t in recovery_times])  # 5帧完美，25帧最差（常数待校准）

# 3. 组合拳连贯性 (0-1)
combo_gaps = calculate_combo_gaps(punches)
flow_score = mean([max(0, 1 - gap / 2) for gap in combo_gaps])  # 0秒完美，2秒最差（常数待校准）

# 4. 护手保持度 (0-1)
guard_violations = count_guard_violations(video_frames)
guard_score = max(0, 1 - guard_violations / total_frames)

# 综合
tech_execution_score = (
    trajectory_score * 0.35 +
    recovery_score * 0.30 +
    flow_score * 0.20 +
    guard_score * 0.15
)
```

#### MVP实现策略

- **可实现**：还原速度、连贯性、护手保持 → MVP上线
- **延后**：轨迹标准性 → v1.1（需开发光流方向分析）

---

### 2.3 维度2：力量表现（25%）

**定义**：单次击打的力量输出质量

#### CV测量指标（按训练类型）

| 训练类型 | 主要指标 | 备选指标 | MVP可行性 |
|-----------|-----------|-----------|-------------|
| 打沙袋 | 出拳速度（简化）| - | ✅ 可实现（复用实战模型速度测量）|
| 打手靶 | 教练身体位移 + 手臂震动 | 出拳末端速度（简化） | ⚠️ 手靶需开发 |
| 空击 | 出拳末端速度（相对值） | 身体旋转幅度 | ✅ 可实现 |

#### 计算公式（以手靶为例，沙袋评估待重构）

```python
# 打手靶：出拳末端速度（简化替代教练位移）
punch_speeds = [p.velocity for p in punches]
power_score = mean(punch_speeds) / MAX_SPEED  # 归一化到0-1（常数待校准）
```

#### MVP实现策略

- **沙袋**：暂不支持（颜色跟踪难度大）→ v1.1或跳过
- **手靶**：用"出拳速度"替代"教练位移" → MVP可用
- **空击**：用"出拳速度" → MVP可用

---

### 2.4 维度3：节奏控制（20%）

**定义**：出拳间隔规律性、组合节奏感

#### CV测量指标

| 指标 | 权重 | 测量方法 | MVP可行性 |
|------|--------|----------|-------------|
| 出拳间隔规律性 | 60% | 相邻出拳时间间隔标准差（越小越稳定） | ✅ 完全可行 |
| 组合内节奏一致性 | 40% | 组合内各拳间隔标准差 | ✅ 完全可行 |

#### 计算公式

```python
# 1. 出拳间隔规律性
punch_intervals = calculate_intervals(punches)
interval_std = std(punch_intervals)
rhythm_regularity = max(0, 1 - interval_std / 2)  # 2秒标准差为最差（常数待校准）

# 2. 组合内节奏一致性
combo_rhythms = []
for combo in combos:
    intra_combo_intervals = calculate_intervals(combo)
    if len(intra_combo_intervals) > 1:
        combo_rhythms.append(std(intra_combo_intervals))
combo_rhythm_score = mean([max(0, 1 - r / 1) for r in combo_rhythms])  # 1秒标准差最差（常数待校准）

rhythm_control_score = rhythm_regularity * 0.6 + combo_rhythm_score * 0.4
```

#### MVP实现策略

- **完全可行**：只需出拳时间戳 → MVP上线

---

### 2.5 维度4：训练强度（15%）

**定义**：整个训练过程中的输出保持度（后段是否掉速？）

#### CV测量指标

| 指标 | 权重 | 测量方法 | MVP可行性 |
|------|--------|----------|-------------|
| 输出密度变化 | 40% | 后半段出拳数 / 前半段出拳数 | ✅ 完全可行 |
| 出拳速度变化 | 40% | 后半段平均速度 / 前半段平均速度 | ✅ 完全可行 |
| 疲劳迹象 | 20% | 后段停顿>3秒次数（反向指标） | ✅ 完全可行 |

#### 计算公式

```python
half_point = video_duration / 2

first_half_punches = count_punches_in_range(punches, 0, half_point)
second_half_punches = count_punches_in_range(punches, half_point, video_duration)
intensity_retention = second_half_punches / first_half_punches
intensity_score = min(intensity_retention, 1)

first_half_speeds = [p.speed for p in punches if p.timestamp < half_point]
second_half_speeds = [p.speed for p in punches if p.timestamp >= half_point]
speed_retention = mean(second_half_speeds) / mean(first_half_speeds)
speed_score = min(speed_retention, 1)

fatigue_signs = count_fatigue_signs(second_half_punches)
fatigue_penalty = min(fatigue_signs / 5, 1)  # 5次疲劳最差（常数待校准）
fatigue_score = 1 - fatigue_penalty

training_intensity_score = (
    intensity_score * 0.4 +
    speed_score * 0.4 +
    fatigue_score * 0.2
)
```

#### MVP实现策略

- **完全可行**：基于出拳时间戳和速度 → MVP上线

---

### 2.6 维度5：技术多样性（10%）

**定义**：使用的拳法类型丰富度、组合变化

#### CV测量指标

| 指标 | 权重 | 测量方法 | MVP可行性 |
|------|--------|----------|-------------|
| 拳法分布均匀度 | 40% | Shannon熵（4种拳法分布） | ✅ 完全可行 |
| 组合拳变化度 | 40% | 不同组合模式数量 | ✅ 完全可行 |
| 非常规拳法使用 | 20% | (勾拳+上勾拳) / 总拳数 | ✅ 完全可行 |

#### 计算公式

```python
punch_type_counts = count_by_type(punches)
punch_type_proportions = normalize(punch_type_counts)
entropy = calculate_shannon_entropy(punch_type_proportions)
# 修复：用实际观察到的拳法数K归一化，而非固定log(4)
K = len(punch_type_counts)  # 实际观察到的拳法数量
max_entropy = log(K) if K > 1 else 1
diversity_score = entropy / max_entropy

combo_patterns = [tuple([p.type for p in combo]) for combo in combos]
unique_combos = len(set(combo_patterns))
combo_variety_score = min(unique_combos / 10, 1)  # 10种组合满分（常数待校准）

unconventional_ratio = (hook_count + uppercut_count) / total_punches
unconventional_score = min(unconventional_ratio / 0.4, 1)  # 40%非常规拳法满分（常数待校准）

tech_diversity_score = (
    diversity_score * 0.4 +
    combo_variety_score * 0.4 +
    unconventional_score * 0.2
)
```

#### MVP实现策略

- **完全可行**：基于出拳类型分类 → MVP上线

---

## 3. 训练类型子模型

### 3.1 为什么需要子模型？

不同训练类型的**评估重点不同**：
- 打手靶：教练控制节奏，高密度≠高质量 → 降低"训练强度"权重
- 打沙袋：力量输出是核心 → 提高"力量表现"权重
- 空击：无接触，测的是"样子" → 提高"技术执行"权重

**方案**：保持5个维度定义不变，只调整权重。

---

### 3.2 子模型权重配置

| 维度 | 打手靶 | 打沙袋 | 空击 |
|-------|---------|---------|-------|
| 技术执行 | 35% ↑ | 20% ↓ | 35% ↑ |
| 力量表现 | 15% ↓ | 35% ↑ | 15% ↓ |
| 节奏控制 | 25% ↑ | 20% | 25% ↑ |
| 训练强度 | 10% ↓ | 15% | 15% |
| 技术多样性 | 15% | 10% ↓ | 10% |

**调整逻辑**：
- 打手靶：技术执行+节奏控制（教练互动质量）
- 打沙袋：力量表现（重拳质量）
- 空击：技术执行+节奏控制（动作标准性）

---

### 3.3 子模型选择逻辑

```
用户输入视频
     ↓
视频类型识别（3种方式，优先级从高到低）：
  1. 文件名包含关键词：["沙袋","bag"] → heavy_bag
  2. 用户手动选择：下拉菜单选择类型
  3. 默认：pad_work（最常见）
     ↓
调用对应子模型评分
```

---

## 4. 评分公式与校准

### 4.1 总分计算公式

```python
def calculate_training_score(metrics, video_type="pad_work"):
    """
    metrics: Dict with all 5 dimension scores (0-1)
    video_type: "pad_work" | "heavy_bag" | "shadow_boxing"
    """
    
    # 子模型权重
    weights = {
        "pad_work": {"tech": 0.35, "power": 0.15, "rhythm": 0.25, "intensity": 0.10, "diversity": 0.15},
        "heavy_bag": {"tech": 0.20, "power": 0.35, "rhythm": 0.20, "intensity": 0.15, "diversity": 0.10},
        "shadow_boxing": {"tech": 0.35, "power": 0.15, "rhythm": 0.25, "intensity": 0.15, "diversity": 0.10}
    }
    w = weights[video_type]
    
    # 加权求和
    raw_score = (
        metrics["tech_execution"] * w["tech"] +
        metrics["power_performance"] * w["power"] +
        metrics["rhythm_control"] * w["rhythm"] +
        metrics["training_intensity"] * w["intensity"] +
        metrics["tech_diversity"] * w["diversity"]
    )
    
    # 映射到0-100分（与实战模型一致）
    final_score = raw_score * 100
    return round(final_score, 1)
```

---

### 4.2 校准方法论

#### 校准目标

- 训练视频评分 **不直接对应比赛能力**
- 但应与**实战评分有弱正相关**（训练质量高的拳手，实战能力通常也高）
- 理想相关系数：r ≈ 0.4-0.6（弱-中度相关）

#### 校准数据需求

| 数据类型 | 数量 | 用途 |
|-----------|--------|------|
| 同一拳手，实战+训练视频对 | 20+对 | 计算相关系数，验证弱正相关 |
| 教练主观评价 | 30+样本 | 对比模型评分 vs 教练打分（目标：r > 0.7） |
| 不同水平拳手训练视频 | 50+样本 | 验证评分区分度（新手<60，进阶60-75，高手>75） |

#### 校准流程（v1.0）

```
Step 1: 收集20+拳手的实战+训练视频对
Step 2: 用当前公式计算训练评分
Step 3: 计算训练分 vs 实战分的相关系数
Step 4: 如果 |r| > 0.7（强相关）或 |r| < 0.3（弱相关），调整权重
Step 5: 迭代至 r ≈ 0.4-0.6
Step 6: 发布v1.0校准版本
```

---

## 5. CV管线需求

### 5.1 新增CV能力（v1.0）

| 能力 | 用途 | 开发优先级 |
|------|------|-------------|
| 沙袋位移跟踪 | 力量表现（沙袋） | P2（v1.1或跳过）|
| 出拳还原速度测量 | 技术执行 | P0（MVP必须） |
| 组合拳识别 | 技术执行+节奏控制 | P0（MVP必须） |
| 疲劳迹象检测 | 训练强度 | P1（MVP可简化） |
| 轨迹方向分析 | 技术执行（精确） | P2（v1.1） |

---

### 5.2 复用现有CV能力（v3.0实战模型）

- ✅ 出拳检测（jab/cross/hook/uppercut分类）
- ✅ 出拳计数
- ✅ 出拳速度测量
- ✅ 拳法类型分布统计

---

## 6. 实施计划

### 6.1 MVP范围（2周）

**目标**：支持打手靶 + 空击视频评分（CV能力可复用实战模型）

| 任务 | 输出 | 负责人 |
|------|--------|--------|
| 创建 `cv_pipeline_pad.py` | 打手靶CV分析脚本 | 拳神 |
| 创建 `cv_pipeline_shadow.py` | 空击CV分析脚本 | 拳神 |
| 创建 `training_scorer.py` | 训练视频评分引擎 | 拳神 |
| 改造 `generate_report_v3.py` | 支持训练视频报告生成 | 拳神 |
| 测试：5个手靶视频 + 3个空击视频 | 验证评分合理性 | 雨叔 |

---

### 6.2 后续迭代

| 版本 | 内容 |
|-------|------|
| v1.0 | 打手靶+空击评分（MVP） |
| v1.1 | 打沙袋评分 |
| v1.2 | 轨迹方向分析（精确技术执行） |
| v1.3 | 教练互动质量评估（手靶专用） |

---

## 7. 版本历史

| 版本 | 维度数 | 核心变更 |
|-------|--------|-----------|
| v1.0 | 5维 | **初始版本**：技术执行/力量表现/节奏控制/训练强度/技术多样性；支持打手靶+空击视频 |

---

## 附录A：与实战模型对比表

| 对比项 | 实战对练模型 v3.0 | 训练视频模型 v1.0 |
|--------|---------------------|---------------------|
| 视频类型 | 实战对练（Sparring） | 打手靶/打沙袋/空击 |
| 评估目标 | 比赛能力 | 训练质量 |
| 维度数量 | 14维 | 5维 |
| 核心维度 | 进攻/防守/反击/战术/体能 | 技术/力量/节奏/强度/多样性 |
| 对抗压力 | ✅ 有 | ❌ 无 |
| 评分相关性 | 基准 | 弱正相关（r≈0.4-0.6） |
| 排名用途 | P4P实战榜 | P4P训练榜 |

---

## 附录B：CV管线文件结构（草案）

```
cv_pipeline/
├── cv_pipeline_sparring.py   # 实战对练CV分析（已有 v3.0）
├── cv_pipeline_bag.py        # 打沙袋CV分析（v1.1）
├── cv_pipeline_pad.py        # 打手靶CV分析（v1.0 MVP）
├── cv_pipeline_shadow.py     # 空击CV分析（v1.0 MVP）
└── common/
    ├── punch_detector.py     # 出拳检测（共用）
    ├── speed_estimator.py    # 速度估计（共用）
    └── bag_tracker.py       # 沙袋跟踪（v1.1）
```

---

> 拳击训练视频评分模型 v1.0 · 完整定义 · 含CV测量方法  
> 与实战对练模型 v3.0 并行使用 · 分别评估训练质量与比赛能力
