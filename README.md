# 零售银行政策与合规 Agentic RAG

面向零售银行政策与合规问答场景的端到端 RAG 系统。

## 项目概览

本项目从监管政策、部门规章和规范性文件中构建可溯源、可评测的问答能力，覆盖文档摄取、混合检索、排序、Agent 编排、模型微调、量化和服务化等完整链路。

示例语料来自国家金融监督管理总局、中国人民银行和政府网站公开发布的监管政策文件。

## 核心能力

- 使用 PyMuPDF 解析 PDF，并支持表格 OCR
- 文档清洗、Markdown 规整、标题/图片/表格关联
- 父子块切分，兼顾检索粒度和生成上下文
- BM25、Dense、Sparse 三路混合召回
- RRF 融合和 Cross-Encoder 精排
- Guardrail、Router、Recovery、Summary、Web Search、Reject 的 Agent 编排
- QA 数据生产、老师模型蒸馏、Pointwise 与 SFT 数据构造
- LoRA 微调、AWQ INT4 量化、vLLM 服务化适配
- 检索与生成质量评测指标

## 架构

```text
PDF
  -> PDF 解析与表格 OCR
  -> 大模型清洗为 Markdown
  -> MongoDB 存储原文与元数据
  -> 父子块切分
  -> Milvus 存储向量

Query
  -> 安全防护
  -> 路由
      -> RAG 问答
      -> 多文档总结
      -> 联网搜索
      -> 拒答
  -> 渐进式自救
  -> 后处理
  -> 答案 + 引用 + 关联图片
```

## 目录结构

```text
compliance-agentic-rag/
  configs/                 # 运行配置
  samples/                 # 公开样例 PDF
  scripts/                 # 可运行示例
  src/compliance_rag/
    adapters/              # PDF、OCR、Mongo、Milvus、内存适配器
    core/                  # 切块、RRF、去重等核心逻辑
    pipelines/             # 数据、训练、评测流水线
    services/              # 检索、Agent、服务入口
  tests/                   # 单元测试
  docs/                    # 架构与状态文档
```

## 快速开始

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/demo_ingest.py
.venv/bin/python scripts/demo_query.py
.venv/bin/python -m unittest discover -s tests -v
```

`demo_ingest.py` 会解析 `samples/` 下的 PDF 并完成分块和向量化；`demo_query.py` 会执行三路召回、RRF 融合和精排，展示最终候选文档。

## 数据与部署

仓库只包含少量公开样例 PDF，不包含私有数据、内部知识库、训练模型权重或密钥信息。

MongoDB、Milvus 和对象存储以 Adapter 形式接入；模型训练、量化和 vLLM 服务化以配置适配器和文档形式提供。

## 评测

当前实现的主要评测指标包括：

- Jaccard 关键词相似度
- 语义答案相似度
- `Recall@K`
- `Hit@K`
- `MRR`

评测模块要求使用带版本标识的金标 QA 数据集，以便每个实验结果都能独立复现。
