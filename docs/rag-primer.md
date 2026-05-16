# RAG Primer — よく聞かれるポイント

実装判断・面接・PR レビューで聞かれがちな論点をまとめたリファレンス。
学術的な網羅性ではなく、**「どこで何を選べばいいか」**を即答できる形を
目指す。各セクション末に agent-kms (本リポ) での選択も併記。

関連: [retrieval-roadmap.md](./retrieval-roadmap.md) — 改善計画一覧。

---

## 1. チャンキング (Chunking)

### 主要戦略

| 戦略 | 説明 | 向く場面 |
|---|---|---|
| **Fixed-size (chars / tokens)** | 一定文字数で機械的に分割 | 構造のないログ、議事録 |
| **Sentence-aware** | 文末で区切る | 短文中心の文書 |
| **Paragraph-aware** | 空行 / 段落で区切る | エッセイ、ブログ |
| **Recursive character split** | LangChain 標準。区切り候補 (`\n\n`, `\n`, `.`, ` `) を優先順にトライ | 雑多な markdown / txt 全般 |
| **Document-structure aware** | 見出し (H1/H2) を境界に使う | 技術ドキュメント、README、ADR |
| **Semantic chunking** | 連続 sentence の embedding 類似度が下がった所で切る | 意味的に均質な大段落を持つ長文 |
| **Sliding window + overlap** | 上記いずれかに 10-20% overlap を加える | 境界跨ぎ情報を失いたくない時 (法律、論文) |
| **Hierarchical (parent-child)** | 小チャンクで検索 → ヒットしたら親段落を返す | 細かい識別子検索 + 広い context が両方必要 |
| **Late chunking** (Jina, 2024) | 文書全体を embedding 後にトークン単位で chunk する。文書全体の文脈が embedding に乗る | 長文 PDF、論文 |
| **Agentic chunking** | LLM が段落の意味境界を判断して分割 | 高コスト許容、最高品質を狙う場合 |

### サイズの基本ルール

- **小さすぎる (< 100 tokens)**: 文脈消失、回答に必要な情報が足りない
- **大きすぎる (> 1500 tokens)**: signal dilution、関連薄い箇所がノイズ化、context window 圧迫
- **典型レンジ**: 200-800 tokens
- **embedding model の max_seq_length に注意**: e5/ruri-v3 系は 512 が安全 (agent-kms は store.py でこれを cap)

### overlap の入れ方

- 0%: シンプル、重複なし、ただし境界で情報が割れるリスク
- 10-20%: 標準。境界跨ぎ救済
- 50%+: index 肥大、最終 distinct chunk が減る (やりすぎ)

### よくある質問への即答

> **Q: チャンクサイズはどう決めればいい?**
> A: クエリの典型長 × 5-10 倍を初期値に。短文 Q&A なら 200 tokens、playbook / 説明系なら 500-800。**評価データで実測してから固定する**。

> **Q: 文書構造 (H2) と固定サイズ、どっちが良い?**
> A: ドキュメントが構造化されているなら H2 分割が圧倒的に有利 (見出しが natural query 表現になる)。雑多なテキストは recursive char split。

> **Q: overlap は必要?**
> A: 質問が「単一段落で完結する」なら不要。「複数段落を読まないと答えられない」なら必要。

### agent-kms の選択
- `chunker.chunk_markdown_h2`: Document-structure aware (H2 分割)。preamble は H1 を heading に
- `chunker.chunk_yaml_per_file`: 1 ファイル = 1 chunk
- **overlap なし**: H2 単位なので構造的に独立した chunk になる前提
- `embed_text = "<file_stem> | <heading>\n\n<body>"`: file stem + heading を embedding に混ぜることで「どの文書のどの section か」を vector に乗せる (recall 改善)

---

## 2. Embedding model 選定

### 主要モデルファミリー

| モデル | 次元 | 強み | 弱み |
|---|---|---|---|
| **OpenAI text-embedding-3-large** | 3072 | 多言語、ベンチ最上位級 | 有料 API、外部送信 |
| **OpenAI text-embedding-3-small** | 1536 | 安い、ベンチ十分 | API |
| **Cohere embed-multilingual-v3** | 1024 | 多言語 | API |
| **intfloat/multilingual-e5-large** | 1024 | 多言語、ローカル、Apache 2.0 | 日本語特化モデルに劣る |
| **intfloat/multilingual-e5-base** | 768 | 上記の軽量版 | 同上 |
| **BAAI/bge-m3** | 1024 | dense + sparse + colbert 一体、多言語 SOTA 近い | 重い |
| **cl-nagoya/ruri-v3-310m** | 768 | **日本語特化、JMTEB 上位**、Apache 2.0 | 英語他言語は普通 |
| **cl-nagoya/ruri-v3-30m** | 256 | 超軽量、ingest 速い | 精度は中位 |
| **pkshatech/GLuCoSE-base-ja-v2** | 768 | 日本語特化、PKSHA | 維持頻度 |

### 選択ガイド

```
コーパスは英語中心? → bge-m3 / e5-large / OpenAI
日本語中心?         → ruri-v3 系
混在?              → multilingual-e5
ローカル必須?       → ruri-v3 / e5 / bge-m3 (API 系を除く)
コスト最重視?       → ruri-v3-30m (256d、ingest 速い)
```

### Passage / Query prefix

E5 系は `passage:` / `query:` prefix を要求 (これを忘れると性能が 10pt 下がることがある)。ruri-v3 は本来 `文章:` / `クエリ:` だが、`passage:` / `query:` でも大幅に劣化しない (実測済)。**agent-kms は両モデル `passage:` / `query:` を使用**。

### よくある質問への即答

> **Q: 次元数は大きい方が良い?**
> A: 必ずしも。`multilingual-e5-base` (768d) と `ruri-v3-310m` (768d) で後者が JP コーパスで +80% F1 改善した実例あり。**次元より model 自体の relevance が支配的**。

> **Q: 多言語 vs 日本語特化、どっち?**
> A: コーパスの 80%+ が日本語なら特化モデル一択。混在 (30%+ 他言語) なら多言語。

> **Q: モデル変えたら何が起きる?**
> A: 全 chunk の vector を再計算 (= 全 ingest 再実行)。**閾値も再キャリブレーション必須** (cosine 分布が違う)。

### agent-kms の選択
- デフォルト `cl-nagoya/ruri-v3-310m` (日本語特化、Apache 2.0、768d、~620 MB)
  — rag-evaluation-jp ベンチで `multilingual-e5-base` 比 F1 +80%
- 英語中心コーパスでは `AGENT_KMS_MODEL=intfloat/multilingual-e5-base` に戻す (broader multilingual、~1 GB)

---

## 3. 検索方式

### 系統比較

| 方式 | 強み | 弱み |
|---|---|---|
| **Dense (embedding)** | 意味的類似、paraphrase | 識別子・固有名詞 (`PrismaClient`) を smear |
| **Sparse (BM25)** | 識別子・厳密一致、解釈可能 | 同義語に弱い |
| **Hybrid (Dense + Sparse)** | 両者の長所、failure rate 大幅削減 | 実装複雑、score 合成が必要 |
| **ColBERT (late interaction)** | token レベル match、高精度 | index サイズ ~30 倍、retrieval 遅い |
| **DPR (Dense Passage Retrieval)** | task 特化 fine-tune 可 | 学習データ必要 |

### Hybrid の合成方法

- **RRF (Reciprocal Rank Fusion)**: `score = Σ 1/(k + rank_i)`。パラメータ 1 個 (`k=60` が定石)、頑健、デファクト
- **Linear combination**: `α × dense_score + β × bm25_score`。要 normalize、要 α/β tune、データに敏感
- **CombSUM / CombMNZ**: 古典手法、現在は RRF が大半

### Reranking

検索 → 上位 N (例 50) → cross-encoder で再 score → 上位 K (例 5) を返す **2 段階**構成。

| Reranker | 特徴 |
|---|---|
| `BAAI/bge-reranker-large` | OSS、多言語、定番 |
| Cohere Rerank | API、多言語、品質高い |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 軽量、英語 |
| LLM rerank | GPT/Claude に「この 50 件を relevance 順に」と聞く、品質高いがコスト |

### よくある質問への即答

> **Q: Hybrid (Dense + BM25) はどれくらい効く?**
> A: Anthropic Contextual Retrieval 報告で 35% → 49% の failure 削減 (Dense only → Dense+BM25)。コーパスと query 種別に依存するが、**識別子検索が混じるなら入れる価値大**。

> **Q: Reranker は必ず必要?**
> A: top-5 の精度が重要なら必須。top-30 を context に入れて LLM に判断させるなら不要なケースも。**コスト + latency と相談**。

> **Q: Dense か Sparse か 1 つ選ぶなら?**
> A: 一般文書なら Dense。コード / API ドキュメントなら Sparse の方が良いことが多い。

### agent-kms の選択
- 現状 **Dense only** (cosine + severity_boost + applicability_penalty + threshold)
- `sparse.py` で BM25 スカフォルド済、wiring は未実装 (roadmap §1)
- Reranker 未実装

---

## 4. クエリ処理

### 主要手法

| 手法 | 何をするか | コスト | 出典 |
|---|---|---|---|
| **Raw** | クエリそのまま embed | 最低 | (baseline) |
| **Query cleaning** | 会話的部分を rule で削る、最終文に絞る等 | 0 (LLM 不要) | 自前実装 |
| **Multi-Query** | LLM で 3-5 個の paraphrase 生成 → 並列検索 → RRF | +1 LLM + N embed | LangChain |
| **HyDE** | LLM で「仮想回答文書」生成 → それを embed → 検索 | +1 LLM | Gao 2022, ACL 2023 |
| **Step-Back** | 具体クエリを抽象クエリに変換 → 両方で検索 | +1 LLM | Zheng 2024, ICLR |
| **RAG-Fusion** | Multi-Query + RRF の production 版 | +1 LLM + N embed | Rackauckas 2024 |
| **Query routing** | LLM が「どの index / source を使うか」を分岐判定 | +1 LLM | (一般 pattern) |
| **Self-RAG** | LLM が「retrieve するか / どう使うか」を判定 reflective | 数 LLM | Asai 2023 |

### 選び方

```
クエリが既に短くて明確 (識別子 / 固有名詞) → Raw で十分
ユーザーの自然言語 prompt 全文 → Query cleaning は最低限必要
語彙ミスマッチ大 (抽象文書) → Step-Back / HyDE
recall 重視 (絶対取りこぼしたくない) → Multi-Query + RRF
複数の index を使い分け → Query routing
```

### よくある質問への即答

> **Q: HyDE と Multi-Query は何が違う?**
> A: Multi-Query は「クエリの言い換え」(質問形態のまま)、HyDE は「回答文書の生成」(declarative)。HyDE の方が文書側との分布合わせが効きやすい。

> **Q: ユーザー prompt 全文を embedding に投げて良い?**
> A: **基本ダメ**。会話 glue (「〜してください」「〜と思います」) と複数 intent が混じって vector が中央に寄り、検索精度が落ちる。最低でも rule-based の clean が必要。

### agent-kms の選択
- 現状 **Raw + template_expand** (固定 keyword 追記のみ)
- Query intent extraction が roadmap §5 として登録 (NEXT 優先度)

---

## 5. Indexing & 保存

### Vector DB 主要選択肢

| DB | 種類 | 長所 | 短所 |
|---|---|---|---|
| **Qdrant** | OSS、Rust | 軽い、metadata filter 強い、hybrid 対応 | スケーラビリティは Milvus 等に劣る |
| **Pinecone** | 商用 SaaS | フルマネージド、すぐ動く | 高コスト、外部送信 |
| **Weaviate** | OSS、Go | GraphQL クエリ、module 豊富 | 設定多い |
| **Milvus** | OSS、Go | 大規模スケール (10M+ vectors) | 運用重い |
| **ChromaDB** | OSS、Python | 開発が一番速い、ローカル | production 機能弱 |
| **pgvector** | Postgres 拡張 | 既存 RDB と統合 | スケール上限 |
| **LanceDB** | OSS、Rust | ファイル DB、組み込み可 | 新興 |
| **FAISS** | Meta、ライブラリ | 高速、研究で定番 | KVS でなく純検索 lib |

### 選択ガイド

```
個人 / 小規模 (< 100k vectors) → Qdrant Docker / ChromaDB / LanceDB
中規模 (< 10M) → Qdrant / Weaviate / pgvector
大規模 (10M+) → Milvus / Vespa
完全マネージド希望 → Pinecone / Weaviate Cloud
既存 Postgres があり統合したい → pgvector
```

### Metadata filtering の重要性

retrieve に **`source_type=session_lesson AND severity=critical`** のような filter を効かせると、recall / precision とも改善することがある。Qdrant / Weaviate は強い。Pinecone も sparse metadata は対応。

### よくある質問への即答

> **Q: ChromaDB と Qdrant、最初はどっち?**
> A: 1 週間で動くものを作るなら Chroma (Python ライブラリだけで完結)。Production 想定なら Qdrant (Docker、metadata filter、hybrid)。

> **Q: pgvector は production で使える?**
> A: 100 万件規模までなら問題ない。それ以上はマネージドの dedicated vector DB の方が安全。

### agent-kms の選択
- **Qdrant** (Docker、bind mount で persistence、metadata filter 活用)

---

## 6. 評価

### Retrieval metrics

| metric | 意味 | 適する場面 |
|---|---|---|
| **Recall@K** | 正解が top-K に何件入っているか | 必須、まず見る |
| **Precision@K** | top-K のうち何件が正解か | top-N 固定検索 |
| **MRR (Mean Reciprocal Rank)** | 最初の正解の rank の逆数の平均 | 1 件 1 個正解の Q&A |
| **NDCG@K** | 順位 + relevance を考慮した ranking 品質 | 学術論文で標準 |
| **F1@T** | threshold-based 検索の P/R 調和平均 | threshold 型 retrieve (agent-kms) |
| **avg_returned** | クエリあたり平均返却数 | context budget の代理指標 |
| **empty rate** | 0 件返るクエリの割合 | threshold 妥当性 |

### End-to-end metrics (LLM 出力後)

| metric | 意味 |
|---|---|
| **Faithfulness** | 生成回答が retrieve context に sticking しているか (Hallucination 防止) |
| **Answer relevance** | 質問に答えているか |
| **Context relevance** | retrieve した context は質問に関連しているか |
| **Context precision** | 関連 context が上位に来ているか |
| **Context recall** | ground truth に必要な情報が context に入っているか |

### 評価フレームワーク

- **Ragas** (Python、人気): faithfulness, answer_relevancy, context_precision/recall
- **TruLens**: production 用 instrumentation + 評価
- **ARES**: 自動評価セット生成
- **MTEB**: embedding model のベンチマークセット
- **BEIR**: retrieval ベンチマーク (英語)
- **JMTEB**: 日本語版 MTEB
- **MIRACL**: 多言語 retrieval ベンチ (日本語含む)

### よくある質問への即答

> **Q: まず何を測ればいい?**
> A: 20-50 件の手作りクエリ + gold chunk のペアを作って **Recall@10, MRR** から。これすらないと「直した」と言い切れない。

> **Q: LLM-as-judge の評価は信用していい?**
> A: 人間評価との相関は task 次第 (0.6-0.9)。スポットチェックで人手と整合性確認しつつ、A/B 比較用には使える。

### agent-kms の選択
- 別 repo [rag-evaluation-jp](https://github.com/spin-glass/rag-evaluation-jp) で手動 eval set 構築 (32 query / 127 chunk)
- 主要 metric: P/R/F1@T, MRR, NDCG@10, avg_returned, empty rate
- threshold 型に合わせた評価設計 (top-K でなく)

---

## 7. Production の関心事

### Latency 予算

```
User query → embedding (10-50ms) → vector search (10-50ms)
  → rerank (50-500ms if used) → LLM generation (1-10s)
```

検索パイプライン全体は 100-500ms が現実的。これ以上だと UX 上「遅い」と感じる。

### Cost 軸

| コスト要因 | 影響 |
|---|---|
| Embedding API 呼び出し | ingest 時に大量 (chunk 数 × ~$0.0001 / call) |
| Vector DB ホスティング | Qdrant Cloud / Pinecone は GB 単価 |
| LLM 生成 | クエリ毎の最大コスト要因、context 長で線形増加 |
| Rerank API | top-50 → top-5 1 回 |

### 鮮度 (freshness)

```
ドキュメント更新頻度
  毎時 → streaming ingest (Kafka → embed → upsert)
  毎日 → cron で batch ingest
  まれ → 手動 agent-kms ingest
```

### 認可・テナント分離

- **per-tenant collection**: 完全分離、ただし管理 overhead
- **per-tenant filter**: 1 collection に metadata filter (`tenant_id`)、漏洩リスク有
- **per-tenant index (Pinecone namespace)**: 中間策

### よくある質問への即答

> **Q: Production で何が一番ヤバい?**
> A: (1) context bloat → LLM コスト爆発、(2) stale data → 信頼性低下、(3) prompt injection via retrieved content。

### agent-kms の選択
- ローカル運用前提なので latency / コストは個人マシン範囲
- 鮮度: Stop hook で session_lesson が自動 upsert、ドキュメント側は手動 `agent-kms ingest`
- テナント分離: 同一 Qdrant の collection 切替で実現 (`japan_ai_org_knowledge` 単一 collection)

---

## 8. よくある故障モード + 診断

| 症状 | 原因候補 | 診断方法 | 対処 |
|---|---|---|---|
| **0 件返る** | threshold が高すぎる、コーパスに該当なし | threshold 下げて hit 数推移を見る、retrieve_simple (boost なし) で確認 | threshold 調整 / コーパス追加 |
| **100+ 件返る** | threshold が低すぎる、クエリが汎用すぎる | クエリ語彙確認、`avg_returned` 計測 | クエリ cleaning / threshold 上げる / max_hits cap |
| **関係ない chunk が上位** | embedding model 不適、クエリと文書の語彙差 | top-10 を目視 + similar embedding 探索 | model 切替 / Contextual Retrieval / HyDE |
| **同じ chunk が複数 source から重複** | ingest source の overlap、dedup なし | source_file 別に payload group | source 整理 / Qdrant filter |
| **更新したドキュメントが反映されない** | re-ingest していない、stable_id 衝突 | Qdrant scroll で last_updated 確認 | `agent-kms ingest --reset` |
| **Lost in the middle** | context 中央の情報が無視される (LLM 共通) | 同じ context を順序入れ替えて test | 重要情報を context 先頭・末尾に寄せる |
| **Hallucination despite context** | LLM が retrieve を無視 | "Per retrieved context: ..." と明示する prompt | prompt 強化 / Self-RAG |
| **Identifier 検索が外す** | dense smear | BM25 で同じクエリ実行して差を見る | Hybrid 化 |
| **抽象クエリが具体的 chunk しか拾わない** | クエリ抽象度と文書抽象度の差 | Step-Back クエリで再検索 | Step-Back / Contextual |
| **threshold チューニングが「効かない」** | embedding 分布が想定外 | score 分布を histogram で可視化 | 別 model 試す |

---

## 9. 用語集 (めいわく略語)

| 略語 | 意味 |
|---|---|
| **RAG** | Retrieval-Augmented Generation |
| **HyDE** | Hypothetical Document Embeddings |
| **RRF** | Reciprocal Rank Fusion |
| **BM25** | Best Matching 25 (Okapi BM25 — sparse 検索の defacto) |
| **TF-IDF** | Term Frequency × Inverse Document Frequency |
| **DPR** | Dense Passage Retrieval (Facebook 2020) |
| **ColBERT** | Contextualized Late Interaction over BERT |
| **MRR** | Mean Reciprocal Rank |
| **NDCG** | Normalized Discounted Cumulative Gain |
| **MTEB** | Massive Text Embedding Benchmark |
| **JMTEB** | Japanese MTEB |
| **MIRACL** | Multilingual Information Retrieval Across a Continuum of Languages |
| **CRAG** | Corrective RAG (Yan 2024) |
| **GraphRAG** | Microsoft の知識グラフ統合 RAG (2024) |
| **SPLADE** | Sparse Lexical AnD Expansion (semantic sparse) |

---

## 10. 推奨スタックの早見表 (2026 春時点)

| シナリオ | スタック |
|---|---|
| **個人で素早く** | ChromaDB + e5-base + LangChain + GPT-4o-mini |
| **個人で日本語特化** | Qdrant + ruri-v3-310m + Gemini Flash (agent-kms 構成) |
| **チーム / 中規模 production** | Qdrant Cloud + bge-m3 + Cohere Rerank + Claude/GPT |
| **エンタープライズ** | Pinecone / Weaviate + text-embedding-3-large + Cohere Rerank + LLM API |
| **完全 on-prem 必須** | Qdrant self-hosted + ruri/bge + vLLM + Llama/Qwen |
| **超大規模 (100M+)** | Milvus + bge-m3 + 分散 reranker |

---

## 参考リソース (深掘り用)

- **Anthropic — Introducing Contextual Retrieval (2024-09)**: hybrid + contextual で 67% failure 削減の事例
- **OpenAI Cookbook — RAG techniques**: 標準的な実装パターン
- **LlamaIndex docs**: 高度パターン (sub-question, hierarchical, summary index) の参照実装
- **LangChain RAG guide**: query expansion / multi-vector のレシピ
- **Pinecone Learn**: 商用視点の良記事多数
- **bclavie/RAGatouille**: ColBERT を簡単に試せる Python lib
- **Ragas**: 評価フレームワーク
- **JMTEB / MIRACL リーダーボード**: 日本語 embedding ベンチ

---

Last updated: 2026-05-15.
