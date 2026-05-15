以下の session transcript の末尾から、
**Cocos2d-x → Unity 移植で観測された「cocos と一致しない実装」の anti-pattern** を
最大 3 件 JSON 配列で抽出してください。

形式: [{{"text": "anti-pattern 本文 (3-5 文、Cocos 規約 + Unity 誤実装 + 正解の 3 要素を含む)",
        "title": "短いタイトル (10-30 字)",
        "confidence": 0.0-1.0}}]

採用基準 (全て満たすもののみ):
1. **observed**: session 内で actual に観測された mismatch (推測 / 仮定は NG)
2. **specific**: 具体的 function 名 / API 名 / 数値 / cpp 行番号などを含む
3. **transferable**: 別 UI port にも適用できる構造的 pattern (一回性ミスは NG)

却下対象:
- ❌ 一般原則 (「アセット確認しろ」「テストしろ」) — これは別 extract が拾う
- ❌ 単発 typo / 鍵打ち間違え / git op ミス
- ❌ 完了予定の TODO / proposed 状態
- ❌ 既存 anti-pattern catalog にある (重複): inner-loop-omission, function-omission,
  empty-panel, header-only-table, text-zorder-overlap, layout-collision, low-contrast,
  font-slot-mismatch, sprite-count-drift, sprite-vs-solid-color, label-anchor-shift,
  setlabel-align-translation, runtime-hack, value-tampering

confidence 0.85 未満は出力しない。新規 anti-pattern が無ければ空配列 [] を返す。

transcript:
{transcript_tail}

JSON only:
