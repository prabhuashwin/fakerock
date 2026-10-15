import threading
import unittest

from mock_bedrock.control import ScenarioError, validate_scenarios
from tests.helpers import Harness

MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
CONVERSE = f"/model/{MODEL}/converse"
STREAM = f"/model/{MODEL}/converse-stream"
BODY = {"messages": [{"role": "user", "content": [{"text": "hi"}]}]}


class SequenceBase(unittest.TestCase):
    startup = None

    def setUp(self):
        self.h = Harness(scenarios=self.startup) if self.startup else Harness()
        self.addCleanup(self.h.stop)

    def put(self, scenarios):
        st, _, _ = self.h.request("/_fakerock/scenarios", scenarios, method="PUT")
        self.assertEqual(st, 200)

    def call(self, path=CONVERSE):
        return self.h.request(path, BODY)

    def statuses(self, n):
        return [self.call()[0] for _ in range(n)]


class TestTimesAndReplies(SequenceBase):
    def test_throttle_twice_then_answer(self):
        self.put([{"id": "t", "times": 2, "reply": {"mode": "throttle"}}])
        self.assertEqual(self.statuses(4), [429, 429, 200, 200])
        rows = self.h.requests_log()
        self.assertEqual([r["mode"] for r in rows], ["scenario", "scenario", "answer", "answer"])
        self.assertEqual(rows[0]["scenario"], "t")
        self.assertIsNone(rows[2]["scenario"])

    def test_fail_the_third_call(self):
        self.put([{"id": "f", "times": 3,
                   "replies": [{"mode": "answer"}, {"mode": "answer"}, {"mode": "throttle"}]}])
        self.assertEqual(self.statuses(4), [200, 200, 429, 200])

    def test_tool_loop(self):
        self.put([{"id": "loop", "times": 3, "replies": [
            {"tool_use": {"name": "search", "input": {"q": "a"}}},
            {"tool_use": {"name": "fetch", "input": {"u": "b"}}},
            {"text": "all done"}]}])
        r1 = self.call()[2]
        r2 = self.call()[2]
        r3 = self.call()[2]
        tu1 = r1["output"]["message"]["content"][0]["toolUse"]
        tu2 = r2["output"]["message"]["content"][0]["toolUse"]
        self.assertEqual((tu1["name"], r1["stopReason"]), ("search", "tool_use"))
        self.assertEqual((tu2["name"], r2["stopReason"]), ("fetch", "tool_use"))
        self.assertEqual(r3["output"]["message"]["content"][0]["text"], "all done")
        self.assertEqual(r3["stopReason"], "end_turn")

    def test_last_reply_repeats_without_times(self):
        self.put([{"replies": [{"text": "one"}, {"text": "two"}]}])
        texts = [self.call()[2]["output"]["message"]["content"][0]["text"] for _ in range(4)]
        self.assertEqual(texts, ["one", "two", "two", "two"])

    def test_falls_through_to_later_scenario(self):
        self.put([{"id": "a", "times": 1, "reply": {"text": "first"}},
                  {"id": "b", "reply": {"text": "second"}}])
        texts = [self.call()[2]["output"]["message"]["content"][0]["text"] for _ in range(3)]
        self.assertEqual(texts, ["first", "second", "second"])


class TestModesInScenarios(SequenceBase):
    def test_stream_error_via_scenario(self):
        self.put([{"id": "se", "times": 1, "reply": {"mode": "stream_error"}}])
        st, _, raw = self.call(STREAM)
        self.assertEqual(st, 200)
        self.assertIn(b"modelStreamErrorException", raw)
        st, _, raw = self.call(STREAM)
        self.assertEqual(st, 200)
        self.assertNotIn(b"modelStreamErrorException", raw)

    def test_stall_honours_stall_seconds(self):
        self.h.set_modes({"_stall_seconds": 0.05})
        self.put([{"times": 1, "reply": {"mode": "stall"}}])
        self.assertEqual(self.statuses(2), [200, 200])
        self.assertEqual([r["mode"] for r in self.h.requests_log()], ["scenario", "answer"])


class TestCounters(SequenceBase):
    startup = [{"id": "s", "times": 1, "reply": {"mode": "throttle"}}]

    def test_reset_clears_counters(self):
        self.assertEqual(self.statuses(2), [429, 200])
        self.h.request("/_fakerock/reset", {})
        self.assertEqual(self.statuses(2), [429, 200])

    def test_set_scenarios_clears_counters(self):
        self.assertEqual(self.statuses(2), [429, 200])
        self.put(self.startup)
        self.assertEqual(self.statuses(2), [429, 200])


class TestValidation(unittest.TestCase):
    def bad(self, sc, text=None):
        with self.assertRaises(ScenarioError) as cm:
            validate_scenarios({"scenarios": [dict(id="x", **sc)]})
        self.assertIn("'x'", str(cm.exception))
        if text:
            self.assertIn(text, str(cm.exception))

    def test_times(self):
        for v in (0, -1, "2", True, 1.5):
            self.bad({"times": v, "reply": {}}, "times")

    def test_replies(self):
        self.bad({"replies": []}, "replies")
        self.bad({"replies": "x"}, "replies")
        self.bad({"replies": [{"text": "a"}, 5]}, "replies[1]")

    def test_both(self):
        self.bad({"reply": {}, "replies": [{}]}, "not both")

    def test_unknown_mode(self):
        self.bad({"reply": {"mode": "explode"}}, "mode")
        self.bad({"replies": [{"mode": 3}]}, "mode")

    def test_valid_and_unknown_keys_allowed(self):
        validate_scenarios({"scenarios": [
            {"id": "ok", "times": 2, "note": "x", "replies": [{"mode": "answer"}, {"text": "t"}]}]})


class TestConcurrency(SequenceBase):
    startup = [{"id": "c", "times": 3, "reply": {"mode": "throttle"}}]

    def test_exactly_times_matches(self):
        out, lock = [], threading.Lock()

        def work():
            st = self.call()[0]
            with lock:
                out.append(st)
        ts = [threading.Thread(target=work) for _ in range(10)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(15)
        self.assertEqual(sorted(out), [200] * 7 + [429] * 3)


if __name__ == "__main__":
    unittest.main()
