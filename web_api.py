from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from agent import app as agent_app

web = FastAPI()

@web.get("/")
def index():
    return HTMLResponse('''<html><body>
<h2>电商知识库 Agent</h2>
<input id="q" placeholder="问问题..." style="width:300px;padding:5px">
<button onclick="ask()" style="padding:5px">问</button>
<pre id="result" style="background:#f5f5f5;padding:15px;margin-top:10px;white-space:pre-wrap;border-radius:5px"></pre>
<script>
function ask() {
    const q = document.getElementById('q').value;
    const result = document.getElementById('result');
    result.innerHTML = '';
    const es = new EventSource('/chat?question=' + encodeURIComponent(q));
    es.onmessage = function(e) {
        if (e.data === '[DONE]') { es.close(); return; }
        result.innerHTML += e.data + '\\n';
    };
    es.onerror = function() { es.close(); };
}
</script>
</body></html>''')

@web.get("/chat")
def chat(question: str):
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
    return StreamingResponse(stream(), media_type="text/event-stream")
