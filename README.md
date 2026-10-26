# 零售银行政策与合规 Agentic RAG 问答系统

面向零售银行政策与合规场景的 Agentic RAG 参考实现。

## 项目说明

原项目从国家金融监管总局、中国人民银行、中国政府网等官方渠道收集了 200+ 篇现行有效监管政策、部门规章和规范性文件，并构建了可溯源、可更新、可评测的问答系统。

本仓库是**开源参考实现**：

- 使用 5 份公开 PDF 作为样例语料
- 不包含私有数据、内部文档或原始训练模型
- 不声称复现原项目的 10w+ QA、95%+ Hit@1 或 93% 综合分
- 目标是完整展示工程方法、代码结构和可运行流程

## 与简历对应关系

| 简历能力点 | 仓库实现 |
| --- | --- |
| 基于 QA 对构建 Pointwise 数据 | `src/compliance_rag/pipelines/qa_factory.py`、`teacher_trace.py` |
| Dense + Sparse + BM25 召回与精排 | `src/compliance_rag/services/retrieval/service.py` |
| Agentic 路由与渐进式自救 | `src/compliance_rag/services/agent/workflow.py` |
| AWQ 量化与 vLLM 推理优化 | `src/compliance_rag/pipelines/training/` |
| 评测指标 | `src/compliance_rag/pipelines/eval/metrics.py` |

## 架构

```text
PDF
  -> PyMuPDF + PaddleOCR PP-StructureV3
  -> LLM Clean
  -> MongoDB
  -> Parent/Child Chunking
  -> Milvus

Query
  -> Guardrail
  -> Router
      -> RAG
      -> Summary
      -> Web Search
      -> Reject
  -> Recovery
  -> Postprocess
```

## 目录

```text
compliance-agentic-rag/
  configs/
  services/
  pipelines/
  src/compliance_rag/
  tests/
  infra/
  docs/
```

## 快速启动

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/demo_ingest.py
.venv/bin/python scripts/demo_query.py
.venv/bin/python -m unittest discover -s tests -v
```

## 目录

```text
compliance-agentic-rag/
  configs/                 # 配置
  samples/                 # 公开样例 PDF
  scripts/                 # 可运行 Demo
  src/compliance_rag/
    adapters/              # PDF、OCR、Mongo、Milvus、内存实现
    core/                  # 切块、RRF、去重等纯逻辑
    pipelines/             # 数据、训练、评测
    services/              # 检索、Agent、服务入口
  tests/                   # 单元测试
  docs/                    # 架构、阶段说明
```

## 诚实声明

原项目的离线评测结果用于说明项目背景，不代表本仓库可以在公开样例上复现相同数字。仓库保留方法实现和评测代码，使用公开数据即可得到可复现的实验结果。
