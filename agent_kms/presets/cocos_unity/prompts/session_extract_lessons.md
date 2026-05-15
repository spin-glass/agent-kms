以下の session transcript の末尾から、
**別 worktree / 別タスクで作業する将来のセッション** が知っていれば
防げた slack を、最大 3 件、JSON 配列で出力してください。

形式: [{{"text": "知見本文 (1-3 文)", "confidence": 0.0-1.0}}]

採用基準 (3 つ全てを満たすもののみ):
1. **再利用性**: 別の作業対象 (別 UI / 別機能 / 別 bug) でも適用される
2. **具体性**: literal 数値, API 名, ファイル種別, 誤った前提など concrete を含む
3. **発生根拠**: session 内で actual 失敗 / 訂正 / 発見を反映 (proposed/予定 は NG)

却下対象 (該当があれば一切出力しない):
- ❌ git/PR/CI/branch の手順 (CLAUDE.md / Constitution に既収載)
- ❌ "X を確認する / Y に気をつける" の vague な advice
- ❌ session 単発のミス (鍵 prefix 打ち間違え, 一回きりの誤操作)
- ❌ 廃棄された実験の実装詳細 (D2Q / hybrid / 廃止 collection 等)
- ❌ プロジェクト内部の評価指標 (Tier1A coverage, MRR, NDCG, ranking 順位)
- ❌ 一回限りの設計判断 (collection 名切替, ファイル削除/移動)
- ❌ 自セッション固有の作業流れ (実装途中のメモ, 次やる予定)

forbidden vocab (skeleton/placeholder/defer/minimal/scope-narrow/dummy/仮実装/今回のみ) 禁止。
**confidence は採用基準への合致度で評価し、0.85 未満は出力しないこと**。
該当事象が無ければ空配列 [] を返すこと。

transcript:
{transcript_tail}

JSON only:
