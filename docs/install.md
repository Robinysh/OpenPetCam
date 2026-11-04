# Install OpenPetCam by hand

`./setup` does all of this for you (see the README). These are the same steps, for when you want to see or change each one.

You need:
- a Thingino-supported pan/tilt camera (reference: Cinnado D1, Ingenic T23N);
- a Linux VM with a public IP running Ubuntu 24.04 (reference: AWS Lightsail `nano_3_0`, $5/month);
- a Mac or Linux computer with `git`, `ssh` and `rsync`.

1. **Flash the camera** with Thingino ([supported cameras and instructions](https://thingino.com)), join it to
   your Wi-Fi and set its root password in the setup portal.
2. **Create the server.** Open TCP 22, 80 and 443, plus UDP 51820 (WireGuard) and UDP 8189 (WebRTC media), in
   the cloud firewall. Note its public IP and SSH key.
3. **Get the latest release and configure it:**
   ```bash
   git clone https://github.com/crimeacs/OpenPetCam.git && cd OpenPetCam
   git checkout "$(git describe --tags --abbrev=0)"     # the latest release, so it can update itself
   cp config/site.env.example config/site.env           # then fill it in
   ```
   Fill in the pet's name, pronoun and description, the server IP, a host name and the time zone. A free host
   name that needs no DNS is `petcam.<ip-with-dashes>.sslip.io`.
4. **Deploy:** `./deploy.sh`. It downloads the models (pinned by SHA-256), installs MediaMTX, Caddy and WireGuard,
   and starts everything. It prints the server's WireGuard public key.
5. **Connect the camera.** In Thingino's WireGuard page fill in address `10.77.0.2/24`, endpoint
   `<server IP>:51820`, peer public key (the key printed in step 4), allowed IPs `10.77.0.1/32` and keepalive
   `25`. Copy the camera's public key into `CAMERA_WG_PUBKEY` in `config/site.env`, then run `./deploy.sh` again.
6. **Set the passwords** on the server, interactively: `ssh -t <server> 'sudo /opt/petcam/set-secrets.sh'`.
   It asks for the web app password and the camera's web login, and optionally a new RTSP login (change the
   default `thingino`/`thingino`).
7. **Open** `https://<PUBLIC_HOST>` on your phone, sign in once, and *Share → Add to Home Screen*. Opened from
   the Home Screen it runs full screen and can send notifications. Invite others from Settings.
