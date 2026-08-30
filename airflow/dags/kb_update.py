from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

dag = DAG(
    'kb_update',
    schedule='0 3 * * *',  # 每日凌晨3点（Airflow 3.x用schedule不用schedule_interval）
    start_date=datetime(2025, 7, 1),
    catchup=False
)

def update_knowledge_base():
    """每日增量更新知识库：检查新文档→分块→Embedding→入库Chroma"""
    import os
    DOCS_DIR = os.path.expanduser("~/ecommerce_rag/data/docs")
    CHROMA_DIR = os.path.expanduser("~/ecommerce_rag/chroma_db")
    # 1. 扫描新文档（对比已入库的文档列表）
    # 2. 新文档分块+Embedding+Chroma.add_documents入库
    # 3. 记录已入库文档列表（避免重复）
    print("知识库增量更新完成")

def run_eval_monitor():
    """跑RAGAS评估监控质量有没有下降"""
    import subprocess
    subprocess.run(["python", os.path.expanduser("~/ecommerce_rag/run_eval.py")], check=True)
    print("评估监控完成")

t1 = PythonOperator(task_id='update_kb', python_callable=update_knowledge_base, dag=dag)
t2 = PythonOperator(task_id='eval_monitor', python_callable=run_eval_monitor, dag=dag)
t1 >> t2
