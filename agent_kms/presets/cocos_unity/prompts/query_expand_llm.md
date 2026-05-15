以下は Cocos2d-x → Unity 移植プロジェクトで AI が UI port 計画時に
発する planning query です。この query が暗黙に含む universal な観点 (asset verification /
faithful port / UI conventions / workflow / completion verification / その他) を query に
明示的に統合した、search retrieve 用の "膨らませた query" を 1 文で返してください。

要件:
- 元 query の topic-specific 情報を全て保持
- universal な観点を 5-8 件、自然言語で list 列挙
- 出力は 1 段落の文字列、200-400 字
- JSON ではなく純粋な文字列
- forbidden vocab (skeleton/placeholder/dummy) は使わない

planning query:
{query}

膨らませた query:
