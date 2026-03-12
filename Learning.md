# Learning Notes - RAG System & Azure Deployment

## RAG (Retrieval-Augmented Generation) とは

- LLMに外部データを検索して渡すことで、正確な回答を生成する手法
- LLM単体の知識に頼らず、最新・独自のデータに基づいた回答が可能

## RAGの基本フロー

1. **データ準備**: テキストデータをチャンクに分割し、embeddingベクトルに変換
2. **検索 (Retrieval)**: ユーザーの質問をembedding化し、類似度の高いチャンクを検索
3. **生成 (Generation)**: 検索結果をコンテキストとしてLLMに渡し、回答を生成

## このプロジェクトで学べること

- OpenAI APIをSDKなしでREST APIとして直接呼び出す方法
- Embeddingベクトルとコサイン類似度による類似検索の仕組み
- FlaskによるシンプルなWebアプリケーション構築
- Azure App Serviceへのデプロイ手順

## Key Concepts

### Embedding

- テキストを数値ベクトルに変換したもの
- 意味が近いテキスト同士はベクトルも近くなる
- OpenAI `text-embedding-3-small` モデルを使用

### Cosine Similarity

- 2つのベクトルの方向の近さを測る指標（-1〜1）
- 1に近いほど類似度が高い
- 計算式: `cos(θ) = A·B / (|A| × |B|)`

### Prompt Engineering in RAG

- 検索結果をsystemプロンプトに埋め込む
- 「以下の情報に基づいて回答してください」というパターン
