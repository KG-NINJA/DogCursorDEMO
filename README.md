# Dog Cursor Control (YOLOv8 + ONNX Runtime Web)

Webブラウザ上で動作する「犬でカーソルを操作する」プロトタイプアプリです。
YOLOv8-nano モデルを ONNX Runtime Web (WASM) で動かし、リアルタイム検出を行います。

## 特徴
- **AI Model**: YOLOv8n (ONNX format)
- **Engine**: ONNX Runtime Web (WebAssembly SIMD multi-threaded)
- **Deployment**: GitHub Pages 対応 (coi-serviceworker 使用)
- **Target**: Intel 10th Gen CPU でのなめらかな動作

## 使い方 (ローカル)
1. フォルダ内でローカルサーバーを起動します。
   ```bash
   python -m http.server
   ```
2. ブラウザで `http://localhost:8000` にアクセスします。
3. カメラの使用を許可すると、犬の認識が始まります。

## 動作モード
このアプリには2つのモードが搭載されています。画面内のボタンで切り替え可能です。

1.  **Precision Mode (YOLOv8)**:
    -   `index.html`
    -   高精度ですが、マシンスペックによっては重くなる場合があります。
    -   ONNX Runtime Web (WASM) 使用。GitHub Pagesでは `coi-serviceworker` が必須です。
2.  **High Speed Mode (COCO-SSD)**:
    -   `index_ssd.html`
    -   非常に高速(60fps)ですが、検出精度はYOLOに劣る場合があります。
    -   TensorFlow.js 使用。

## GitHub Pages へのデプロイ方法
このリポジトリは GitHub Pages で即座に動作するように構成されています。

1. GitHub にリポジトリを作成し、このフォルダの中身を全てプッシュします。
2. GitHub リポジトリの **Settings > Pages** を開きます。
3. **Source** を `Deploy from a branch` にし、`main` (または `master`) ブランチの `/ (root)` を指定して保存します。
4. 数分後、発行されたURLにアクセスしてください。

### トラブルシューティング
GitHub Pages で動かない場合、画面下の「黒いコンソール領域」を確認してください。

- **`ReferenceError: SharedArrayBuffer is not defined`**:
    - `coi-serviceworker.js` が正しく読み込まれていません。リロードしてみてください。
    - それでもダメな場合、ブラウザのキャッシュをクリアするか、プライベートウィンドウで試してください。
- **`404 Not Found` (onnx)**:
    - `yolov8n.onnx` ファイルが正しくアップロードされているか確認してください（Git LFSが必要なサイズではありませんが、除外されていないか注意）。
- **カメラが起動しない**:
    - ブラウザの権限設定でカメラを許可してください。
    - HTTPS (GitHub Pages標準) または localhost でのみ動作します。

## ファイル構成
- `index.html`: YOLO版メインUI
- `index_ssd.html`: COCO-SSD版メインUI
- `script.js`: YOLO推論ロジック
- `script_ssd.js`: COCO-SSD推論ロジック
- `coi-serviceworker.js`: CORS回避用スクリプト
- `yolov8n.onnx`: 推論モデル
