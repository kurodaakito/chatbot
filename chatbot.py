
# %%
# 必要なライブラリをインポートします
import openai  # OpenAI APIを利用するためのライブラリ
from openai import OpenAI # OpenAIクライアントを直接インポート
import os      # 環境変数を操作するためのライブラリ
import gradio as gr # Webインターフェースを簡単に作成するためのライブラリ
from dotenv import load_dotenv # .envファイルから環境変数を読み込むためのライブラリ
# %%

# .envファイルから環境変数を読み込み
# これにより、APIキーなどの機密情報をコードに直接書き込むのを防ぐ
load_dotenv()

# OpenAIクライアントを初期化
# APIキーは環境変数 "OPEN_API_KEY" から取得
client = OpenAI(api_key=os.environ.get("OPEN_API_KEY"))
# client = OpenAI() # APIキーが環境変数OPENAI_API_KEYに設定されていれば、引数なしでも動作する場合あり

# openaiライブラリのグローバルなAPIキーも設定
openai.api_key = os.getenv("OPEN_API_KEY")

# チャットボットのメイン機能を定義する関数
def chatbot(prompt):
    # OpenAIのChatCompletion APIを利用して、応答を生成
    response = client.chat.completions.create(
        model="gpt-4.1",  # 使用するGPTモデル
        messages=[{"role": "user", "content": prompt}] # ユーザーからのプロンプトをメッセージとして渡す
    )
    # APIからの応答メッセージの内容（ボットの返答）を返す
    return response.choices[0].message.content

# Gradioインターフェースを作成します
iface = gr.Interface(
    fn=chatbot,  # インターフェースが呼び出す関数（チャットボット機能）
    inputs=gr.Textbox(lines=2, label="あなたのメッセージ"),  # 入力コンポーネント（複数行のテキストボックス）
    outputs=gr.Textbox(label="Botの応答"),  # 出力コンポーネント（テキストボックス）
    title="OpenAI Chatbot",  # インターフェースのタイトル
    description="OpenAI APIを使ったシンプルなチャットボット"  # インターフェースの説明
)

# ChromaDBライブラリをインポート
import chromadb

# ChromaDBクライアントを初期化
# Client()はデフォルトでインメモリデータベースを作成、永続化設定も可能
chroma_client = chromadb.Client()

# "my_collectionsannzanntohuruame" という名前で新しいコレクションを作成
# コレクションは、ドキュメントとその埋め込みベクトルを格納する場所
collection = chroma_client.create_collection(name="my_collectionsannzanntohuruame")

# JSONファイルを扱うためのjsonライブラリをインポート
import json

# 読み込むデータファイル（JSONL形式）のパスを指定
data_path = "aozorabunko-dedupe-clean.jsonl" # スペースを削除して修正

# データファイルを開いて読み込む
with open(data_path, "r", encoding="utf-8") as f: # encoding="utf-8" の追加を推奨
    data = f.readlines() # ファイルの各行をリストとして読み込む

# 読み込んだ各行（JSON文字列）をPythonの辞書オブジェクトに変換
data = [json.loads(x) for x in data]

# 読み込んだデータの件数を表示
print(f"読み込んだデータの件数: {len(data)}")

# 読み込んだデータをChromaDBのコレクションに追加
collection.add(
    documents=[x["text"] for x in data],  # 各データエントリの "text" フィールドをドキュメントとして追加
    ids=[x["meta"]["作品ID"] for x in data]  # 各データエントリの "meta" -> "作品ID" フィールドをドキュメントの一意なIDとして追加
)

print("ChromaDBへのデータ追加が完了しました。")



# 検索クエリを定義
query_text = "吾輩は猫である"

# コレクションから類似するドキュメントを検索
# n_resultsは取得する結果の数
results = collection.query(
    query_texts=[query_text],
    n_results=2
)

print("\n検索結果:")
for i, (doc, id_val) in enumerate(zip(results['documents'][0], results['ids'][0])):
    print(f"--- 検索結果 {i+1} ---")
    print(f"ID: {id_val}")
    print(f"内容の一部: {doc[:200]}...") # ドキュメントの冒頭200文字を表示

# 検索機能とチャットボットを組み合わせる例
def hybrid_chatbot(prompt):
    # まずChromaDBで関連する情報を検索
    retrieved_results = collection.query(
        query_texts=[prompt],
        n_results=1 # 関連性の高いドキュメントを1つ取得
    )

    context = ""
    if retrieved_results and retrieved_results['documents'] and retrieved_results['documents'][0]:
        context = retrieved_results['documents'][0][0] # 取得したドキュメントの内容

    # 取得した情報をOpenAIのプロンプトに組み込む
    if context:
        full_prompt = f"以下の情報に基づいて質問に答えてください。もし情報が不足している場合は、その旨を伝えてください。\n\n情報: {context}\n\n質問: {prompt}"
    else:
        full_prompt = prompt # 関連情報がない場合は元のプロンプトを使用

    # OpenAIのChatCompletion APIを利用して、応答を生成
    response = client.chat.completions.create(
        model="gpt-4",  # 使用するGPTモデル (gpt-4.1は存在しないためgpt-4に修正)
        messages=[{"role": "user", "content": full_prompt}]
    )
    return response.choices[0].message.content

# Gradioインターフェースを更新（ハイブリッドチャットボット機能に）
iface_hybrid = gr.Interface(
    fn=hybrid_chatbot,
    inputs=gr.Textbox(lines=2, label="あなたのメッセージ"),
    outputs=gr.Textbox(label="Botの応答"),
    title="OpenAI & ChromaDB ハイブリッドチャットボット",
    description="ChromaDBで青空文庫のデータを参照し、OpenAI APIで応答を生成するチャットボット"
)

# スクリプトが直接実行された場合にGradioインターフェースを起動
if __name__ == "__main__":
    iface_hybrid.launch()