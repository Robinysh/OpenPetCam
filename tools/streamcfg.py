"""Read or change the camera's prudynt stream settings via its local API (port 8080)."""
import sys, json, time, urllib.request
sys.path.insert(0, "/opt/petcam")
import app

app.cam_login()
key = json.loads(urllib.request.urlopen(urllib.request.Request("http://10.77.0.2/x/api-key.cgi", headers={"Cookie": app._cam["cookie"]}), timeout=10).read())["api_key"]


def cfg(body):
    req = urllib.request.Request("http://10.77.0.2:8080/api/v1/config", data=json.dumps(body, separators=(",", ":")).encode(),
                                 headers={"Content-Type": "application/json", "X-API-Key": key}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=15).read())


FIELDS = {"stream0": {"mode": None, "bitrate": None, "fps": None, "gop": None, "max_gop": None}}
cmd = sys.argv[1]
if cmd == "show":
    print(json.dumps(cfg(FIELDS)))
elif cmd == "set":
    mode, bitrate, fps = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    print("set ->", json.dumps(cfg({"stream0": {"mode": mode, "bitrate": bitrate, "fps": fps, "gop": fps * 2, "max_gop": fps * 4}})))
    time.sleep(4)
    print("now ->", json.dumps(cfg(FIELDS)))
elif cmd == "save":
    print("save ->", json.dumps(cfg({"action": {"save_config": None}})))
