# オンラインサービング（APIサーバー）での中間出力対応

## 現在の状態

### ✅ 実装済み
**オフライン使用** (`LLM.generate()`)
```python
from vllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-VL-8B-Instruct")
params = SamplingParams(output_hidden_states=True, output_logits=True)
outputs = llm.generate("prompt", params)  # ✓ 動く
```

### ⏳ 未実装（追加作業が必要）
**オンラインサービング** (APIサーバー経由)
```bash
# これを動作させるには追加実装が必要
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "prompt": "What is AI?",
    "output_hidden_states": true,
    "output_logits": true
  }'
```

---

## 必要な実装

### Phase 2: APIサーバー対応

#### 1. CompletionRequest の拡張

**ファイル:** `vllm/entrypoints/openai/completion/protocol.py`

```python
class CompletionRequest(OpenAIBaseModel):
    # ... 既存のフィールド ...

    # 追加: 中間出力パラメータ
    output_hidden_states: bool | Literal["final", "all"] = Field(
        default=False,
        description="Whether to return hidden states from the model"
    )
    output_logits: bool = Field(
        default=False,
        description="Whether to return pre-sampling logits"
    )
    output_attention_weights: bool = Field(
        default=False,
        description="Whether to return attention weights (future)"
    )
```

#### 2. CompletionResponseChoice の拡張

**ファイル:** `vllm/entrypoints/openai/responses/protocol.py`

```python
class CompletionResponseChoice(OpenAIBaseModel):
    # ... 既存のフィールド ...

    # 追加: 中間出力（base64エンコード or 別エンドポイント）
    hidden_states: str | None = Field(
        default=None,
        description="Base64-encoded hidden states tensor"
    )
    logits: str | None = Field(
        default=None,
        description="Base64-encoded logits tensor"
    )

    # オプション: テンソルのメタデータ
    hidden_states_shape: list[int] | None = None
    logits_shape: list[int] | None = None
```

#### 3. SamplingParams への変換

**ファイル:** `vllm/entrypoints/openai/completion/handler.py` (または類似)

```python
def _get_sampling_params(request: CompletionRequest) -> SamplingParams:
    return SamplingParams(
        # ... 既存のマッピング ...

        # 追加
        output_hidden_states=request.output_hidden_states,
        output_logits=request.output_logits,
        output_attention_weights=request.output_attention_weights,
    )
```

#### 4. レスポンスの構築

```python
def _create_completion_response(
    output: RequestOutput,
    request: CompletionRequest
) -> CompletionResponse:
    choices = []
    for completion in output.outputs:
        choice_data = CompletionResponseChoice(
            # ... 既存のフィールド ...
        )

        # 中間出力の追加
        if completion.hidden_states is not None:
            choice_data.hidden_states = _encode_tensor(completion.hidden_states)
            choice_data.hidden_states_shape = list(completion.hidden_states.shape)

        if completion.logits is not None:
            choice_data.logits = _encode_tensor(completion.logits)
            choice_data.logits_shape = list(completion.logits.shape)

        choices.append(choice_data)

    return CompletionResponse(choices=choices, ...)
```

---

## 実装の課題と解決策

### 課題1: テンソルのシリアライゼーション

**問題:** `torch.Tensor` はJSONに直接変換できない

**解決策A: Base64エンコード** (推奨)
```python
import base64
import io
import torch

def _encode_tensor(tensor: torch.Tensor) -> str:
    """Encode tensor to base64 string"""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def _decode_tensor(encoded: str) -> torch.Tensor:
    """Decode base64 string to tensor"""
    buffer = io.BytesIO(base64.b64decode(encoded))
    return torch.load(buffer)
```

**解決策B: 別エンドポイント**
```python
# /v1/completions/features/{request_id} でテンソルを取得
# メインレスポンスにはメタデータのみ
{
  "hidden_states_url": "/v1/completions/features/req-123/hidden",
  "logits_url": "/v1/completions/features/req-123/logits"
}
```

**解決策C: NumPy配列** (軽量だが精度損失の可能性)
```python
def _tensor_to_list(tensor: torch.Tensor) -> list:
    return tensor.cpu().numpy().tolist()
```

### 課題2: レスポンスサイズ

**問題:** Logitsが大きすぎる（50トークン × 152K語彙 ≈ 30MB）

**解決策:**
1. **圧縮:**
   ```python
   import gzip
   compressed = gzip.compress(tensor_bytes)
   encoded = base64.b64encode(compressed)
   ```

2. **ストリーミング:**
   ```python
   # Server-Sent Events (SSE) でチャンク送信
   yield f"data: {chunk}\n\n"
   ```

3. **オプトイン設計:**
   ```json
   {
     "output_logits": true,
     "logits_compression": "gzip",  // 圧縮を明示的に要求
     "max_response_size_mb": 100     // サイズ制限
   }
   ```

### 課題3: OpenAI API互換性

**問題:** OpenAIの公式APIには存在しないフィールド

**解決策: 拡張フィールドを使う**
```json
{
  "choices": [...],
  "usage": {...},

  "vllm_extensions": {
    "intermediate_outputs": {
      "hidden_states": "base64...",
      "logits": "base64..."
    }
  }
}
```

---

## 使用例（実装後）

### 基本的な使用

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-8B-Instruct",
    "prompt": "What is machine learning?",
    "max_tokens": 50,
    "output_hidden_states": true,
    "output_logits": true
  }'
```

**レスポンス:**
```json
{
  "id": "cmpl-xxx",
  "object": "text_completion",
  "created": 1234567890,
  "model": "Qwen/Qwen3-VL-8B-Instruct",
  "choices": [
    {
      "text": "Machine learning is...",
      "index": 0,
      "finish_reason": "stop",
      "hidden_states": "base64encodeddata...",
      "hidden_states_shape": [50, 4096],
      "logits": "base64encodeddata...",
      "logits_shape": [50, 152064]
    }
  ]
}
```

### Python クライアント

```python
import requests
import base64
import io
import torch

# リクエスト
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "Qwen/Qwen3-VL-8B-Instruct",
        "prompt": "What is AI?",
        "output_hidden_states": True,
        "output_logits": True
    }
)

data = response.json()
choice = data["choices"][0]

# デコード
def decode_tensor(encoded_str: str) -> torch.Tensor:
    buffer = io.BytesIO(base64.b64decode(encoded_str))
    return torch.load(buffer)

hidden_states = decode_tensor(choice["hidden_states"])
logits = decode_tensor(choice["logits"])

print(f"Hidden states: {hidden_states.shape}")
print(f"Logits: {logits.shape}")
```

---

## 実装の優先順位

### Phase 2.1: 最小限の実装（1-2日）
- [ ] `CompletionRequest` にパラメータ追加
- [ ] `SamplingParams` への変換
- [ ] Base64エンコード実装
- [ ] レスポンスに含める
- [ ] 基本的なテスト

### Phase 2.2: 最適化（2-3日）
- [ ] 圧縮サポート
- [ ] サイズ制限
- [ ] エラーハンドリング
- [ ] ストリーミング対応（オプション）

### Phase 2.3: 高度な機能（1週間）
- [ ] 別エンドポイント実装
- [ ] キャッシュ機構
- [ ] Chat Completions対応
- [ ] WebSocketサポート

---

## ファイル一覧（実装対象）

### 必須
1. **`vllm/entrypoints/openai/completion/protocol.py`**
   - `CompletionRequest` の拡張

2. **`vllm/entrypoints/openai/responses/protocol.py`**
   - `CompletionResponseChoice` の拡張

3. **`vllm/entrypoints/openai/completion/handler.py`** (または類似ファイル)
   - パラメータ変換
   - レスポンス構築

### オプション（Chat Completions対応）
4. **`vllm/entrypoints/openai/chat_completion/protocol.py`**
5. **`vllm/entrypoints/openai/chat_completion/handler.py`**

---

## テスト

### 単体テスト

```python
# tests/entrypoints/openai/test_completion_intermediate.py
def test_completion_with_hidden_states():
    request = {
        "model": "facebook/opt-125m",
        "prompt": "Hello",
        "output_hidden_states": True
    }

    response = client.post("/v1/completions", json=request)
    assert response.status_code == 200

    data = response.json()
    choice = data["choices"][0]
    assert "hidden_states" in choice
    assert choice["hidden_states_shape"] is not None
```

### 統合テスト

```bash
# 起動
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --port 8000

# テスト
curl http://localhost:8000/v1/completions \
  -d '{"prompt": "Test", "output_hidden_states": true}' | jq
```

---

## 代替案: gRPC / WebSocket

JSONの制限を避けるため、バイナリプロトコルも検討可能：

### gRPC
```protobuf
message CompletionResponse {
  repeated Choice choices = 1;

  message Choice {
    string text = 1;
    bytes hidden_states = 2;  // 直接バイナリ
    bytes logits = 3;
  }
}
```

### WebSocket
```python
# クライアント
async with websockets.connect("ws://localhost:8000/v1/stream") as ws:
    await ws.send(json.dumps(request))

    # テキストとテンソルを別々に受信
    text = await ws.recv()  # JSON
    hidden_states = await ws.recv()  # バイナリ
    logits = await ws.recv()  # バイナリ
```

---

## まとめ

| 方法 | 実装難易度 | 互換性 | パフォーマンス |
|------|-----------|--------|---------------|
| **Base64 in JSON** | ⭐⭐☆☆☆ | ✅ 高 | ⚠️ 中（サイズ大） |
| **別エンドポイント** | ⭐⭐⭐☆☆ | ✅ 高 | ✅ 高 |
| **gRPC** | ⭐⭐⭐⭐☆ | ⚠️ 低 | ✅ 最高 |
| **WebSocket** | ⭐⭐⭐☆☆ | ⚠️ 中 | ✅ 高 |

**推奨:** まずBase64エンコードで実装し、必要に応じて別エンドポイントを追加

---

## 次のステップ

1. Phase 2.1の最小実装を完了
2. 動作確認とテスト
3. ドキュメント更新
4. Phase 2.2で最適化
5. 必要に応じてPhase 2.3の高度な機能を追加

**現在のステータス:** Phase 1完了 → Phase 2準備完了
