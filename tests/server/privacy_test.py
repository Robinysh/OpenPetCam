"""Toggle Thingino privacy mode on/off via prudynt's API and check the picture recovers."""
import sys, json, time, urllib.request, hashlib
sys.path.insert(0, "/opt/petcam")
import app

app.cam_login()
key = json.loads(urllib.request.urlopen(urllib.request.Request("http://10.77.0.2/x/api-key.cgi", headers={"Cookie": app._cam["cookie"]}), timeout=10).read())["api_key"]


def api(body):
    req = urllib.request.Request("http://10.77.0.2:8080/api/v1/config", data=json.dumps(body, separators=(",", ":")).encode(),
                                 headers={"Content-Type": "application/json", "X-API-Key": key}, method="POST")
    return json.loads(urllib.request.urlopen(req, timeout=15).read() or b"{}")


def snap(tag):
    b = urllib.request.urlopen(urllib.request.Request("http://10.77.0.2/x/ch1.jpg", headers={"Cookie": app._cam["cookie"]}), timeout=10).read()
    open(f"/tmp/priv_{tag}.jpg", "wb").write(b)
    return hashlib.md5(b).hexdigest()[:8], len(b)


print("privacy now:", api({"privacy": {"enabled": None}}))
print("before :", snap("before"))
print("on     :", api({"privacy": {"enabled": True}}))
time.sleep(5)
print("during :", snap("during"))
print("off    :", api({"privacy": {"enabled": False}}))
time.sleep(5)
a = snap("after1"); time.sleep(3); b = snap("after2")
print("after  :", a, b, "-> picture", "LIVE (changing)" if a[0] != b[0] else "IDENTICAL (frozen?)")
