from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent.execution.trace import RuntimeTrace


class TraceObservabilityScenarioTestCase(unittest.TestCase):
    def test_runtime_trace_supports_debug_json_dump_and_assertion_lookup(self) -> None:
        trace = RuntimeTrace()
        trace.add("event", "received", payload={"timestamp": 10, "type": "user_text_input"})
        trace.add("guard", "filtered", findings=[{"allowed": False, "reason": "cooldown active"}])

        self.assertEqual(trace.stages(), ("event", "guard"))
        self.assertEqual(trace.find("guard")[0].payload["findings"][0]["reason"], "cooldown active")
        self.assertIn("01 event:received", trace.to_debug_string())

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "trace.json"
            trace.dump_json(path)
            dumped = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(dumped["events"][1]["stage"], "guard")


if __name__ == "__main__":
    unittest.main()
