"""Automated tests for P2 features: Local RAG, One-Click Model Manager, and Portable Packaging."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from thai_offline_ai.agent.config import AgentSettings
from thai_offline_ai.agent.orchestrator import AgentOrchestrator
from thai_offline_ai.agent.schemas import OperatingMode
from thai_offline_ai.cli import main as cli_main
from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.llm_types import LLMResult
from thai_offline_ai.local_rag import LocalRAGStore, chunk_text
from thai_offline_ai.model_manager import ModelManager
from thai_offline_ai.packaging import build_portable_bundle
from thai_offline_ai.ui import build_app


def test_local_rag_ingest_search_and_grounded_qa(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    settings.llm.enabled = True

    class StubLLM:
        def chat(self, messages) -> LLMResult:
            assert "[Doc #1" in messages[-1].content
            return LLMResult(
                text="กรมธรรม์คุ้มครองอุบัติเหตุสูงสุด 500,000 บาท [Doc #1]",
                backend="ollama",
                model="stub-rag-llm",
            )

    store = LocalRAGStore(tmp_path / "rag.db", settings=settings, llm_client=StubLLM())
    doc_file = tmp_path / "policy.md"
    doc_file.write_text(
        "# คู่มือกรมธรรม์ประกันภัย\n\nกรมธรรม์ชั้นหนึ่งคุ้มครองอุบัติเหตุสูงสุด 500,000 บาท และต่ออายุผ่านแอปได้ทันที",
        encoding="utf-8",
    )
    other_file = tmp_path / "travel.txt"
    other_file.write_text("แนะนำที่เที่ยวเชียงใหม่ ดอยอินทนนท์ และแม่กำปอง", encoding="utf-8")

    ingested = store.ingest_files([doc_file, other_file])
    assert len(ingested) == 2
    assert len(store.list_documents()) == 2

    matches = store.search("คุ้มครองอุบัติเหตุเท่าไหร่", top_k=2)
    assert matches and matches[0].title == "policy.md"
    assert "500,000" in matches[0].text

    ans = store.ask("วงเงินคุ้มครองอุบัติเหตุเท่าไหร่", use_llm=True)
    assert ans.used_llm is True
    assert "500,000" in ans.answer and "[Doc #1]" in ans.answer

    ext = store.ask("เชียงใหม่มีที่ไหนบ้าง", use_llm=False)
    assert ext.used_llm is False
    assert "ดอยอินทนนท์" in ext.answer

    assert store.delete_document(ingested[0].doc_id) is True
    assert len(store.list_documents()) == 1
    store.clear_all()
    assert store.list_documents() == []
    assert len(chunk_text("กขค " * 400, chunk_chars=200, overlap_chars=40)) >= 4


def test_jarvis_offline_mode_can_use_local_rag_search_tool(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    rag = LocalRAGStore(tmp_path / "rag.db", settings=settings)
    rag.ingest_text("Company Handbook", "วันลาพักร้อนประจำปีได้สูงสุด 15 วันทำการ")

    replies = [
        json.dumps(
            {
                "action": "tool_call",
                "plan_summary": "ค้นหาข้อมูลวันลาพักร้อนจาก Local RAG",
                "tool_name": "rag_search",
                "arguments": {"query": "วันลาพักร้อน", "top_k": 2},
            }
        ),
        json.dumps(
            {
                "action": "final_answer",
                "plan_summary": "สรุปจากเอกสารภายในเครื่อง",
                "final_answer": "จากเอกสาร Company Handbook คุณมีสิทธิ์ลาพักร้อนได้สูงสุด 15 วันทำการครับ",
            }
        ),
    ]

    class ScriptedLLM:
        def chat(self, messages) -> LLMResult:
            return LLMResult(text=replies.pop(0), backend="ollama", model="stub")

    agent_cfg = AgentSettings()
    agent_cfg.memory.db_path = str(tmp_path / "mem.db")
    orch = AgentOrchestrator(
        settings, agent_cfg, llm_client=ScriptedLLM(), rag_store=rag
    )
    resp = orch.run_turn("s-rag", "เช็กสิทธิ์วันลาพักร้อนให้หน่อย", mode=OperatingMode.OFFLINE)
    assert resp.status == "completed"
    assert "15 วัน" in resp.answer
    assert resp.tool_history[0].tool_name == "rag_search"
    assert resp.tool_history[0].ok is True


def test_model_manager_blocks_offline_and_runs_setup_when_authorized(tmp_path, monkeypatch):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    script_dir = tmp_path / "scripts"
    script_dir.mkdir(parents=True)
    (script_dir / "setup_kokoro_tts_model.py").write_text("print('installed ok')\n", encoding="utf-8")

    mgr = ModelManager(settings, project_root=tmp_path)
    statuses = mgr.list_statuses()
    assert len(statuses) == 6
    assert len(mgr.to_table_rows()) == 6

    # Must block when allow_setup_network is False
    with pytest.raises(WorkbenchError, match="Setup Mode"):
        mgr.run_setup("kokoro_thai_onnx", allow_setup_network=False)

    monkeypatch.setattr(
        "subprocess.run",
        lambda cmd, **kwargs: SimpleNamespace(returncode=0, stdout="[OK] Kokoro ready", stderr=""),
    )
    res = mgr.run_setup("kokoro_thai_onnx", allow_setup_network=True)
    assert res["key"] == "kokoro_thai_onnx"
    assert "[OK] Kokoro ready" in res["output"]


def test_portable_packaging_and_cli_p2_commands(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "run_app.py").write_text("# app\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# readme\n", encoding="utf-8")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("app:\n  output_dir: outputs\n", encoding="utf-8")

    settings = Settings(app=AppConfig(output_dir=str(tmp_path / "outputs")))
    bundle = build_portable_bundle(settings, project_root=tmp_path)
    assert (tmp_path / "outputs" / "portable_bundle" / "start_workbench.bat").is_file()
    assert (tmp_path / "outputs" / "portable_bundle" / "portable_manifest.json").is_file()
    assert "run_app.py" in bundle.copied_files

    assert cli_main(["--config", str(cfg), "models"]) == 0
    models_out = json.loads(capsys.readouterr().out)
    assert len(models_out["models"]) == 6

    doc = tmp_path / "note.txt"
    doc.write_text("รหัสโครงการออฟไลน์คือ THAI-LOCAL-2026", encoding="utf-8")
    assert (
        cli_main(
            [
                "--config",
                str(cfg),
                "rag",
                "--ingest",
                str(doc),
                "--ask",
                "รหัสโครงการคืออะไร",
                "--no-llm",
            ]
        )
        == 0
    )
    rag_out = json.loads(capsys.readouterr().out)
    assert "THAI-LOCAL-2026" in rag_out["answer"]["answer"]

    assert cli_main(["--config", str(cfg), "package-portable"]) == 0
    pkg_out = json.loads(capsys.readouterr().out)
    assert pkg_out["launcher_bat"].endswith("start_workbench.bat")


def test_p2_ui_callbacks_for_rag_and_model_manager(tmp_path):
    settings = Settings(app=AppConfig(output_dir=str(tmp_path)))
    demo, _ = build_app(settings)
    try:
        ingest_fn = next(k for k, v in demo.fns.items() if v.api_name == "rag_ingest")
        res_ing = asyncio.run(
            demo.process_api(
                ingest_fn,
                inputs=[None, "คู่มือทดสอบ", "ระบบรองรับการทำงานบน CPU 100% แบบออฟไลน์"],
            )
        )
        assert "นำเข้าสำเร็จ 1 เอกสาร" in res_ing["data"][1]

        ask_fn = next(k for k, v in demo.fns.items() if v.api_name == "rag_ask")
        res_ask = asyncio.run(
            demo.process_api(ask_fn, inputs=["ระบบทำงานบนอะไร", False])
        )
        assert "CPU 100%" in res_ask["data"][0]

        pkg_fn = next(k for k, v in demo.fns.items() if v.api_name == "package_portable")
        res_pkg = asyncio.run(demo.process_api(pkg_fn, inputs=[]))
        assert "Portable Desktop Bundle" in res_pkg["data"][0]
    finally:
        demo.close()
