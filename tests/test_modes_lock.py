import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from mock_bedrock import control as control_mod
from mock_bedrock.control import Control


class StubPack:
    def mode_key(self, call, family):
        return "x"

    def default_mode(self, family):
        return "answer"


class ModesLockTest(unittest.TestCase):
    def test_concurrent_reload(self):
        # smoke test for concurrent reloads (cannot prove absence of races)
        with tempfile.TemporaryDirectory() as d, mock.patch.object(control_mod, "warn"):
            ctl = Control(d)
            modes = Path(d) / "modes.json"
            valid = [
                json.dumps({"chat": {"_default": "answer"}}),
                json.dumps({"chat": {"_default": "throttle"}}),
            ]
            modes.write_text(valid[0], encoding="utf-8")
            pack, errors, seen = StubPack(), [], set()

            def worker():
                try:
                    for _ in range(200):
                        seen.add(ctl.mode_for(pack, None, "chat"))
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            i = 0
            while any(t.is_alive() for t in threads):
                text = "{not json" if i % 7 == 3 else valid[i % 2]
                try:
                    modes.write_text(text, encoding="utf-8")
                except OSError:
                    pass
                i += 1
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            self.assertTrue(seen <= {"answer", "throttle"}, seen)
            modes.write_text(json.dumps({"chat": {"_default": "throttle"}}), encoding="utf-8")
            self.assertEqual(ctl.mode_for(pack, None, "chat"), "throttle")


if __name__ == "__main__":
    unittest.main()
