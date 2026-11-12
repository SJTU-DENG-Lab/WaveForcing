"""Control-plane integration with a bounded in-memory generation worker."""
import unittest

from fastapi.testclient import TestClient
from interactive_fakes import FakeRuntime, wait_until
from wave_rt.interactive.server import Controller, make_app
from wave_rt.interactive.state import Limits


class HttpTest(unittest.TestCase):
    def test_attach_prompt_and_playback_ack(self):
        self.exercise("")

    def test_notebook_proxy_and_reload(self):
        self.exercise("/absproxy/8890")

    def test_live_stream_advances_without_played(self):
        # TestClient reads sync StreamingResponse to completion, so drive the
        # live path through take_live (same code the /live generator calls) and
        # only assert that anonymous /live is rejected.
        runtime = FakeRuntime()
        controller = Controller(runtime, "old", Limits())
        try:
            wait_until(lambda: controller.session.end == 3)
            token = controller.attach()["client"]
            data = controller.take_live(0, controller.session_id, token, timeout=2)
            self.assertEqual(data, b"old")
            self.assertEqual(controller.session.pace, 0)
            # Delivery pace alone asks for the next buffered chunk; /played stays -1.
            wait_until(lambda: controller.session.end >= 4)
            self.assertEqual(controller.session.played, -1)
            with TestClient(make_app(controller)) as client:
                self.assertEqual(client.get("/health").json()["transport"], "h264-fmp4-live")
                self.assertEqual(client.get("/live?after=-1").status_code, 409)
        finally:
            controller.close()
        self.assertTrue(runtime.closed)

    def exercise(self, prefix):
        runtime = FakeRuntime()
        controller = Controller(runtime, "old", Limits())
        with TestClient(make_app(controller), base_url="http://testserver" + prefix) as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.get("/health").json()["service"], "wavert-interactive")
            token = client.post("/attach").json()["client"]
            headers = {"X-Client-Id": token, "X-Session-Id": controller.session_id}
            self.assertEqual(client.post("/attach", headers=headers).json()["client"], token)
            self.assertEqual(client.post("/attach").status_code, 409)
            self.assertEqual(client.get("/state").status_code, 409)
            wait_until(lambda: client.get("/state", headers=headers).json()["end"] == 3)
            state = client.get("/state", headers=headers).json()
            self.assertEqual((state["fps"], state["start_buffer"], state["active"]), (16, 3, 0))
            self.assertIsNone(state["offer"])
            self.assertEqual(client.post("/prompt", headers=headers, json={"prompt": "new"}).status_code, 200)
            self.assertTrue(client.get("/state", headers=headers).json()["pending"])
            # The committed chunks keep the old prompt; playback pulls in the new one.
            played = client.post("/played", headers=headers, json={"branch": 0, "chunk": 0})
            self.assertEqual(played.status_code, 200)
            wait_until(lambda: not client.get("/state", headers=headers).json()["pending"])
            self.assertEqual(client.get("/chunk/0/2", headers=headers).content, b"old")
            media = client.get("/chunk/0/3", headers=headers)
            self.assertEqual(media.headers["content-type"], "video/mp4")
            self.assertEqual(media.content, b"new")
            self.assertEqual(client.get("/chunk/0/0", headers=headers).status_code, 404)
            self.assertEqual(client.get("/chunk/0/4", headers=headers).status_code, 404)
            self.assertEqual(client.get("/chunk/1/3", headers=headers).status_code, 404)
            for body in ({"branch": 0, "chunk": 9999}, {"branch": 1, "chunk": 1}):
                self.assertEqual(client.post("/played", headers=headers, json=body).status_code, 409)
            # Browser diagnostics are accepted from the attached viewer only.
            batch = {"page": "p", "events": [{"event": "waiting", "seq": 1, "data": {"counted": True}}]}
            self.assertEqual(client.post("/trace", json=batch).status_code, 409)
            accepted = client.post("/trace", headers=headers, json=batch).json()
            self.assertEqual((accepted["accepted"], accepted["writer_error"]), (1, None))
            self.assertEqual(client.post("/trace", headers=headers, json={"events": "x"}).status_code, 409)
            self.assertEqual(client.get("/trace?source=client").json()["rows"][0]["row"]["seq"], 1)
            server_rows = client.get("/trace?limit=1000").json()["rows"]
            self.assertEqual([row["chunk"] for row in server_rows if row["event"] == "chunk"], [0, 1, 2, 3])
            self.assertEqual([row["start"] for row in server_rows if row["event"] == "switch"], [3])
            self.assertEqual(client.post("/detach", headers=headers).status_code, 200)
            self.assertEqual(client.post("/attach").status_code, 200)
        self.assertTrue(runtime.closed)


if __name__ == "__main__":
    unittest.main()
