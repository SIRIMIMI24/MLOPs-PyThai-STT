"""Phase 7 — Conversation Intelligence tests (stub LLM; no models, mic, or network)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from thai_offline_ai.cli import main as cli_main
from thai_offline_ai.config import AppConfig, Settings
from thai_offline_ai.conversation_intel import (
    ConversationIntelligenceService,
    Utterance,
    find_domain_terms,
    load_utterances_from_json,
    parse_speaker_text,
)
from thai_offline_ai.errors import WorkbenchError
from thai_offline_ai.llm_types import LLMResult
from thai_offline_ai.ui import build_app

DIALOG = (
    "SPEAKER_00: สวัสดีครับ ขอสอบถามเรื่องต่ออายุกรมธรรม์\n"
    "SPEAKER_01: ได้ครับ เดี๋ยวส่งเอกสารต่ออายุให้ภายในวันศุกร์\n"
    "SPEAKER_00: ตกลงครับ แล้วจะชำระเบี้ยประกันผ่านแอป"
)


@dataclass
class StubLLM:
    replies: list[str]
    calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def chat(self, messages) -> LLMResult:
        self.calls += 1
        self.prompts.append(messages[-1].content)
        return LLMResult(text=self.replies.pop(0), backend="ollama", model="stub-llm")


def _settings(tmp_path) -> Settings:
    return Settings(app=AppConfig(output_dir=str(tmp_path)))


def test_parse_and_deterministic_analytics_without_llm(tmp_path):
    utts = parse_speaker_text(DIALOG + "\nบรรทัดที่ไม่มีผู้พูด")
    assert [u.speaker for u in utts] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00", None]
    report = ConversationIntelligenceService(_settings(tmp_path)).analyze(
        utts, domain_terms=["กรมธรรม์", "เบี้ยประกัน", "ไม่มีคำนี้"], use_llm=False
    )
    assert report.summary is None and report.analysis_error is None
    assert report.speakers == ["SPEAKER_00", "SPEAKER_01"]
    assert {s.speaker for s in report.speaker_stats} == {"SPEAKER_00", "SPEAKER_01", "UNATTRIBUTED"}
    assert abs(sum(s.word_share for s in report.speaker_stats) - 1.0) < 1e-9
    assert sorted(d["term"] for d in report.domain_terms) == ["กรมธรรม์", "เบี้ยประกัน"]
    assert "no business roles" in report.note


def test_llm_items_are_grounded_and_ungrounded_are_dropped(tmp_path):
    reply = json.dumps(
        {
            "summary": "ลูกค้าสอบถามต่ออายุกรมธรรม์ และตกลงชำระผ่านแอป",
            "topics": ["ต่ออายุ", "ชำระเบี้ย"],
            "action_items": [
                {
                    "description": "ส่งเอกสารต่ออายุ",
                    "owner_speaker": "SPEAKER_01",
                    "due": "วันศุกร์",
                    "evidence_index": 1,
                },
                {"description": "ผีสาง", "owner_speaker": "SPEAKER_09", "evidence_index": 99},
                {"description": "ชำระเบี้ย", "owner_speaker": "Agent", "evidence_index": 2},
            ],
            "decisions": [{"description": "ชำระผ่านแอป", "evidence_index": 2}],
        },
        ensure_ascii=False,
    )
    llm = StubLLM(["not json at all", f"```json\n{reply}\n```"])
    report = ConversationIntelligenceService(_settings(tmp_path), llm_client=llm).analyze(
        parse_speaker_text(DIALOG)
    )
    assert llm.calls == 2  # one repair retry
    assert "<<<TRANSCRIPT>>>" in llm.prompts[0]
    assert report.summary and report.llm_model == "stub-llm"
    assert report.dropped_ungrounded == 1
    first = report.action_items[0]
    assert first.evidence_text == "ได้ครับ เดี๋ยวส่งเอกสารต่ออายุให้ภายในวันศุกร์"  # copied from transcript
    assert first.owner_speaker == "SPEAKER_01" and first.due == "วันศุกร์"
    assert report.action_items[1].owner_speaker is None  # invented role rejected
    assert report.decisions[0].evidence_speaker == "SPEAKER_00"
    assert "Action Items" in report.to_markdown()


def test_llm_failure_keeps_deterministic_results(tmp_path):
    llm = StubLLM(["bad", "still bad", "nope"])
    report = ConversationIntelligenceService(_settings(tmp_path), llm_client=llm).analyze(
        parse_speaker_text(DIALOG)
    )
    assert report.summary is None and report.analysis_error
    assert report.speaker_stats and report.keywords
    with pytest.raises(WorkbenchError):
        ConversationIntelligenceService(_settings(tmp_path)).analyze([], use_llm=False)


def test_long_transcript_is_truncated_flagged(tmp_path):
    utts = [Utterance(i, "SPEAKER_00", "ทดสอบข้อความยาวมาก " * 5) for i in range(40)]
    llm = StubLLM([json.dumps({"summary": "ok", "topics": [], "action_items": [], "decisions": []})])
    svc = ConversationIntelligenceService(_settings(tmp_path), llm_client=llm, max_prompt_chars=500)
    assert svc.analyze(utts).truncated is True


def test_find_domain_terms_counts_and_indexes():
    utts = [Utterance(0, None, "เบี้ยประกัน และ เบี้ยประกัน"), Utterance(1, None, "ไม่เกี่ยว")]
    assert find_domain_terms(utts, ["เบี้ยประกัน"]) == [
        {"term": "เบี้ยประกัน", "count": 2, "utterance_indexes": [0]}
    ]


def test_cli_conversation_intel_from_transcript_json_no_llm(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("app:\n  output_dir: outputs\n", encoding="utf-8")
    tj = tmp_path / "t.json"
    tj.write_text(
        json.dumps(
            {
                "segments": [
                    {"text": "สวัสดีครับ", "start": 0.0, "end": 1.5, "speaker": "SPEAKER_00"},
                    {"text": "ยินดีครับ", "start": 1.5, "end": 3.0, "speaker": "SPEAKER_01"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    assert len(load_utterances_from_json(tj)) == 2
    rc = cli_main(["--config", str(cfg), "conversation-intel", "--transcript-json", str(tj), "--no-llm"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["utterance_count"] == 2
    assert out["speaker_stats"][0]["talk_seconds"] == 1.5
    assert (tmp_path / "outputs" / "conversation_intel").is_dir()
    assert cli_main(["--config", str(cfg), "conversation-intel"]) == 2


def test_ui_conversation_intel_callback_with_pasted_text(tmp_path):
    demo, _ = build_app(_settings(tmp_path))
    try:
        fn = next(k for k, v in demo.fns.items() if v.api_name == "conversation_intel")
        res = asyncio.run(demo.process_api(fn, inputs=[None, DIALOG, "กรมธรรม์", False]))
        assert "สัดส่วนการพูด" in res["data"][0]
        payload = res["data"][1]
        payload = getattr(payload, "root", payload)
        assert payload["utterance_count"] == 3
        assert "วิเคราะห์ 3 ช่วงคำพูด" in res["data"][2]
    finally:
        demo.close()
