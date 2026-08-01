# boxingai · 拳击评分模型

拳神系统的量化评分模型文档集。数据驱动，不靠感觉评分。

## 模型体系

| 模型 | 适用场景 | 说明 |
|---|---|---|
| `BOXING_SCORING_MODEL.md` | 拳击对练/技术评估 | 14 维三层评分框架（技术执行 55% + 基础体能 45%） |
| `BAG_SCORING_MODEL.md` | 沙袋训练评估 | 沙袋出拳的量化评分维度 |
| `SHADOW_SCORING_MODEL.md` | 空击评估 | 空击训练的技术评分 |
| `TRAINING_SCORING_MODEL.md` | 训练投入评估 | 训练量/强度/频次评分 |
| `P4P_RANKING_MODEL.md` | 馆内排名 | 技术水平 50% / 训练投入 25% / 进步速度 15% / 活跃度 10%，金牌≥70 / 白银≥55 / 青铜≥35 / 钢铁<35 |

## 相关 schema

- `schemas/metrics_schema.json` — 指标数据结构
- `schemas/scores_schema.json` — 评分数据结构

## 使用

评分管线将视频分析产出的量化指标，按模型映射为各维度得分并合成总分。模型迭代记录见各文档历史。
