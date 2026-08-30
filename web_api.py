import os
import time
import threading
from collections import defaultdict

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

from agent import app as agent_app

# 读取项目根目录的 .env（本地开发用；服务器上用 --env-file 注入）
load_dotenv()

# 访问口令：从环境变量 DEMO_KEY 读取，不写死在代码里
DEMO_KEY = os.environ.get("DEMO_KEY", "")

app = FastAPI()

# ---------- 简易限流：同一个 IP 60 秒内最多 10 次请求 ----------
_hits = defaultdict(list)
_lock = threading.Lock()

def _too_fast(ip: str, limit: int = 10, window: int = 60) -> bool:
    now = time.time()
    with _lock:
        recent = [t for t in _hits[ip] if now - t < window]
        recent.append(now)
        _hits[ip] = recent
        return len(recent) > limit

@app.get("/health")
def health():
    """健康检查：给监控用，一条命令确认服务活着"""
    return {"status": "ok"}

@app.get("/")
def index():
    return HTMLResponse('''<html><head><meta charset="utf-8"></head><body>
<h2>电商知识库 Agent</h2>
<p style="color:#666;font-size:13px">访问口令请向作者索取</p>
<input id="key" placeholder="访问口令" style="width:200px;padding:5px;margin-right:8px">
<input id="q" placeholder="问问题，例如：退货政策是什么" style="width:320px;padding:5px">
<button onclick="ask()" style="padding:5px 12px">问</button>
<pre id="result" style="background:#f5f5f5;padding:15px;margin-top:10px;white-space:pre-wrap;border-radius:5px;min-height:60px"></pre>
<script>
function ask() {
    const q = document.getElementById('q').value;
    const key = document.getElementById('key').value;
    const result = document.getElementById('result');
    if (!key) { result.textContent = '请先输入访问口令'; return; }
    result.textContent = '思考中...';
    const es = new EventSource('/chat?question=' + encodeURIComponent(q) + '&key=' + encodeURIComponent(key));
    es.onmessage = function(e) {
        if (result.textContent === '思考中...') { result.textContent = ''; }
        if (e.data === '[DONE]') { es.close(); return; }
        result.textContent += e.data + '\\n';
    };
    es.onerror = function() {
        if (result.textContent === '思考中...') {
            result.textContent = '连接失败：请检查口令是否正确，或稍后再试';
        }
        es.close();
    };
}
</script>
</body></html>''')

@app.get("/chat")
def chat(question: str, key: str = "", request: Request = None):
    # 第一关：限流（放在口令校验之前，暴力试口令的请求也会被限速）
    client_ip = request.client.host if request and request.client else "unknown"
    if _too_fast(client_ip):
        return JSONResponse({"error": "请求太频繁，请 1 分钟后再试"}, status_code=429)
    # 第二关：口令校验
    if not DEMO_KEY:
        return JSONResponse({"error": "服务器未配置 DEMO_KEY，请检查 .env"}, status_code=500)
    if key != DEMO_KEY:
        return JSONResponse({"error": "访问口令错误"}, status_code=403)

    def stream():
        initial = {"question": question, "tool": "", "tool_results": [], "answer": "", "needs_retry": False, "retry_count": 0, "memory_hit": False}
        yield f"data: 开始处理：{question}\n\n"
        final = {}
        for output in agent_app.stream(initial):
            for node, data in output.items():
                final.update(data)
                if node == "classify":
                    if data.get("memory_hit"):
                        yield "data: 🧠 命中记忆，直接返回\n\n"
                    else:
                        yield f"data: 🎯 意图分类 -> {data.get('tool','')}\n\n"
                elif node == "rag":
                    yield "data: 📄 RAG检索完成\n\n"
                elif node == "calculator":
                    yield "data: 🔢 计算完成\n\n"
                elif node == "web_search":
                    yield "data: 🌐 Web搜索完成（模拟）\n\n"
                elif node == "generate":
                    yield "data: ✍️ 生成完成\n\n"
                elif node == "reflect":
                    status = "PASS ✅" if not data.get("needs_retry") else "FAIL ❌"
                    yield f"data: 🔍 自检 {status}\n\n"
        yield f"data: ✅ 答案：{final.get('answer','')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
