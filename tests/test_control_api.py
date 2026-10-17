import json
import unittest
import urllib.error
import urllib.request

from mock_bedrock.testing import FakerockClient, MockBedrock
from tests.helpers import Harness

CONVERSE = {"messages": [{"role": "user", "content": [{"text": "hello"}]}]}
SC = {"scenarios": [{"id": "boom", "match": {"prompt": "hello"},
                     "reply": {"status": 418, "error_type": "Teapot", "message": "short"}}]}
START_SC = [{"id": "start", "reply": {"text": "from-startup"}}]


class ControlApiTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness(scenarios=START_SC)
        self.addCleanup(self.h.stop)

    def converse(self):
        return self.h.request("/model/m/converse", CONVERSE)

    def test_scenarios_take_effect_and_count(self):
        s, _, b = self.h.request("/_fakerock/scenarios", SC, method="PUT")
        self.assertEqual((s, b), (200, {"ok": True, "count": 1}))
        self.assertEqual(self.converse()[0], 418)
        s, _, _ = self.h.request("/_fakerock/scenarios", SC["scenarios"])  # bare list, POST
        self.assertEqual(s, 200)

    def test_invalid_scenarios_400(self):
        bad = {"scenarios": [{"id": "bad-one", "match": {"prompt": "("}}]}
        s, h, b = self.h.request("/_fakerock/scenarios", bad)
        self.assertEqual(s, 400)
        self.assertEqual(h["x-amzn-ErrorType"], "ValidationException")
        self.assertIn("bad-one", b["message"])
        self.assertEqual(self.converse()[0], 200)  # startup scenario still active

    def test_non_string_reply_fields_rejected_and_old_scenarios_kept(self):
        bad_replies = [
            {"reply": {"text": None}},
            {"replies": [{"text": 5}]},
            {"reply": {"message": None}},
            {"reply": {"stop_reason": 1}},
            {"reply": {"error_type": 3}},
            {"reply": {"tool_use": {"name": "x", "id": ["a"]}}},
            {"reply": {"tool_use": {"name": "x", "id": ""}}},
        ]
        for bad in bad_replies:
            with self.subTest(bad=bad):
                s, _, _ = self.h.request("/_fakerock/scenarios", {"scenarios": [bad]}, method="PUT")
                self.assertEqual(s, 400)
                s, _, b = self.converse()
                self.assertEqual(s, 200)
                self.assertIn("from-startup", json.dumps(b))

    def test_bad_json_400(self):
        req = urllib.request.Request(self.h.base + "/_fakerock/scenarios", data=b"{nope", method="POST")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 400)

    def test_modes_override_and_reset(self):
        h = Harness()
        self.addCleanup(h.stop)
        h.set_modes({"chat": {"_default": "unavailable"}})
        self.assertEqual(h.request("/model/m/converse", CONVERSE)[0], 503)
        s, _, b = h.request("/_fakerock/modes", {"chat": {"_default": "throttle"}}, method="PUT")
        self.assertEqual((s, b), (200, {"ok": True}))
        self.assertEqual(h.request("/model/m/converse", CONVERSE)[0], 429)
        self.assertEqual(h.request("/_fakerock/modes", {"chat": "x"})[0], 400)
        self.assertEqual(h.request("/model/m/converse", CONVERSE)[0], 429)
        self.assertEqual(h.request("/_fakerock/reset")[0], 200)
        self.assertEqual(h.request("/model/m/converse", CONVERSE)[0], 503)

    def test_reset_restores_scenarios_and_clears_requests(self):
        self.h.request("/_fakerock/scenarios", SC)
        self.assertEqual(self.converse()[0], 418)
        b = self.h.request("/_fakerock/requests", method="GET")[2]
        self.assertEqual(len(b["requests"]), 1)
        self.assertEqual(b["requests"][0]["api"], "converse")
        self.assertEqual(self.h.request("/_fakerock/reset")[2], {"ok": True})
        self.assertEqual(self.h.request("/_fakerock/requests", method="GET")[2], {"requests": []})
        s, _, b = self.converse()
        self.assertEqual(s, 200)
        self.assertIn("from-startup", json.dumps(b))

    def test_control_calls_not_logged(self):
        self.h.request("/_fakerock/scenarios", SC)
        self.h.request("/_fakerock/modes", {})
        self.h.request("/_fakerock/requests", method="GET")
        self.h.request("/_fakerock/reset")
        self.assertEqual(self.h.requests_log(), [])
        self.assertEqual(self.h.request("/_fakerock/requests", method="GET")[2], {"requests": []})

    def test_unknown_and_wrong_method(self):
        s, h, _ = self.h.request("/_fakerock/nope", {})
        self.assertEqual(s, 404)
        self.assertEqual(h["x-amzn-ErrorType"], "UnknownOperationException")
        s, h, _ = self.h.request("/_fakerock/reset", method="GET")
        self.assertEqual(s, 405)
        self.assertEqual(h["x-amzn-ErrorType"], "MethodNotAllowed")
        self.assertEqual(self.h.request("/_fakerock/requests", {})[0], 405)

    def test_existing_routes_unchanged(self):
        s, _, b = self.h.request("/health", method="GET")
        self.assertEqual((s, b["ok"]), (200, True))
        self.assertEqual(self.h.request("/elsewhere", {})[0], 404)
        self.assertEqual(self.h.request("/elsewhere", method="GET")[0], 404)


class ClientTests(unittest.TestCase):
    def test_client_against_separate_server(self):
        h = Harness()
        self.addCleanup(h.stop)
        c = FakerockClient(h.base)
        self.assertIsNone(c.last_request())
        c.set_modes({"chat": {"_default": "throttle"}})
        self.assertEqual(h.request("/model/a.b/converse", CONVERSE)[0], 429)
        c.set_modes({})
        h.request("/model/other/converse", CONVERSE)
        h.request("/model/a.b/invoke", {"anthropic_version": "x", "max_tokens": 5,
                                        "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(len(c.requests()), 3)
        self.assertEqual(len(c.requests(api="converse")), 2)
        self.assertEqual(len(c.requests(model="a.b")), 2)
        self.assertEqual(len(c.requests(api="converse", model="a.b")), 1)
        self.assertEqual(c.last_request()["api"], "invoke")
        c.set_scenarios(SC)
        with self.assertRaises(RuntimeError) as cm:
            c.set_scenarios({"scenarios": [{"id": "zzz", "match": {"model": "("}}]})
        self.assertIn("zzz", str(cm.exception))
        c.reset()
        self.assertEqual(c.requests(), [])

    def test_mockbedrock_delegates(self):
        with MockBedrock(modes={"chat": {"_default": "unavailable"}}) as mock:
            def post():
                req = urllib.request.Request(mock.endpoint_url + "/model/m/converse",
                                             data=json.dumps(CONVERSE).encode(), method="POST")
                try:
                    return urllib.request.urlopen(req).status
                except urllib.error.HTTPError as e:
                    return e.code
            self.assertEqual(post(), 503)
            self.assertEqual(mock.last_request()["status"], 503)
            mock.set_modes({})
            self.assertEqual(post(), 200)
            mock.reset()
            self.assertEqual(post(), 503)
            self.assertEqual(len(mock.requests()), 1)
            mock.set_scenarios(SC)
            self.assertEqual(post(), 418)


if __name__ == "__main__":
    unittest.main()
