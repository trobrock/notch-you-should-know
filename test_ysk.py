import importlib.util
import json
import pathlib
import unittest

MODULE = pathlib.Path(__file__).parent / "extensions" / "you-should-know" / "ysk.py"
SPEC = importlib.util.spec_from_file_location("ysk", MODULE)
ysk = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(ysk)


class YouShouldKnowTests(unittest.TestCase):
    def test_transcript_is_text_only_and_bounded(self):
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image", "data": "secret"}]},
            {"role": "assistant", "content": [{"type": "thinking", "text": "hidden"}, {"type": "text", "text": "x" * 25_000}]},
            {"role": "assistant", "content": [{"type": "tool_use", "name": "bash", "arguments": {"command": "git diff --check"}}]},
            {"role": "user", "content": [{"type": "tool_result", "text": ""}]},
            {"role": "user", "content": [{"type": "tool_result", "text": "tests failed", "is_error": True}]},
        ]
        transcript = ysk.transcript_of(messages)
        self.assertLessEqual(len(transcript), ysk.MAX_TRANSCRIPT)
        self.assertNotIn("hidden", transcript)
        self.assertNotIn("secret", transcript)
        self.assertIn('Tool call bash: {"command":"git diff --check"}', transcript)
        self.assertIn("Tool result (succeeded): (no output)", transcript)
        self.assertIn("Tool result (failed): tests failed", transcript)
        self.assertTrue(transcript.endswith("tests failed"))

    def test_parse_notch_events_returns_final_text_and_usage(self):
        output = "\n".join(
            [
                json.dumps({"type": "text_delta", "text": "ignore"}),
                json.dumps(
                    {
                        "type": "turn_end",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": "warning"}]},
                        "usage": {"cost_usd": 0.001},
                    }
                ),
            ]
        )
        text, usage = ysk.parse_notch_events(output)
        self.assertEqual(text, "warning")
        self.assertEqual(usage["cost_usd"], 0.001)

    def test_gate_validation_rejects_incomplete_probabilities(self):
        valid = {"type": "choice", "choice": "warn", "confidence": 0.9, "probabilities": {"warn": 0.9, "quiet": 0.1}}
        self.assertTrue(ysk.valid_choice(valid, ["warn", "quiet"]))
        self.assertFalse(ysk.finite_probability(True))
        valid["probabilities"].pop("quiet")
        self.assertFalse(ysk.valid_choice(valid, ["warn", "quiet"]))

    def test_sanitize_note_handles_none_and_controls(self):
        self.assertEqual(ysk.sanitize_note("NONE."), "")
        self.assertEqual(ysk.sanitize_note("bad\x00thing"), "bad thing")

    def test_unqualified_explore_model_uses_current_provider(self):
        observer = ysk.Observer(ysk.RPC())
        observer.provider = "openai"
        observer.model = "main"
        observer.explore_model = "gpt-5.6-luna"
        self.assertEqual(observer.selected_model(), "openai/gpt-5.6-luna")
        observer.message_end({"provider": "anthropic", "model": "next", "message": {"role": "assistant", "content": []}})
        self.assertEqual(observer.selected_model(), "anthropic/gpt-5.6-luna")

    def test_explanation_child_uses_only_user_settings(self):
        observer = ysk.Observer(ysk.RPC())
        observer.executable = "/notch"
        observer.provider = "openai"
        observer.model = "gpt-5.6-luna"
        captured = {}

        def execute(command, args, timeout):
            captured.update(command=command, args=args, timeout=timeout)
            event = {
                "type": "turn_end",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "NONE"}]},
                "usage": {},
            }
            return {"stdout": json.dumps(event)}

        observer.host_exec = execute
        text, _ = observer.run_model("snapshot", "", 7)
        self.assertEqual(text, "NONE")
        self.assertEqual(captured["command"], "/notch")
        self.assertEqual(captured["timeout"], 7)
        args = captured["args"]
        self.assertEqual(args[args.index("--setting-sources") + 1], "user")
        self.assertIn("--no-tools", args)
        self.assertIn("--no-extensions", args)
        self.assertIn("--max-turns", args)

    def test_empty_durable_state_does_not_warn(self):
        class EmptyRPC:
            def __init__(self):
                self.calls = []

            def host(self, method, params, timeout=50):
                self.calls.append((method, params))
                if method == "host.session.entries":
                    return None  # Older Notch versions encoded an empty slice as null.
                return None

        rpc = EmptyRPC()
        observer = ysk.Observer(rpc)
        observer.session_id = "session-1"
        observer.restore()
        self.assertEqual(
            [method for method, _ in rpc.calls].count("host.session.entries"),
            3,
        )
        self.assertNotIn("host.ui.notify", [method for method, _ in rpc.calls])

    def test_session_append_is_bound_to_restored_session(self):
        class RecordingRPC:
            def __init__(self):
                self.calls = []

            def host(self, method, params, timeout=50):
                self.calls.append((method, params))
                return None

        rpc = RecordingRPC()
        observer = ysk.Observer(rpc)
        observer.session_id = "session-1"
        observer.append("state", {"enabled": True})
        self.assertEqual(rpc.calls[0][1]["session_id"], "session-1")


if __name__ == "__main__":
    unittest.main()
