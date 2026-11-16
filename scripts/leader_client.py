#!/usr/bin/env python3
"""Stream an SO-101 leader arm to the Isaac teleop running on the cluster.

Run this on the computer the leader arm is plugged into (your laptop). It reads the six
STS3215 servos over USB and posts their positions to the teleop page's URL, which VS Code
already forwards to the cluster. The sim turns them into joint angles with the
calibration files in calibration/ (see xle_leisaac/teleop/leader_arm.py).

Needs only Python 3 and pyserial (`pip install pyserial`; a LeRobot env already has it).

    python leader_client.py "http://localhost:22874/?token=XYZ" --port /dev/ttyACM0
    python leader_client.py "http://localhost:22874/?token=XYZ" --port COM5 --arm left
    python leader_client.py "http://localhost:22874/?token=XYZ" --fake     # no arm: a slow wave

Port names: Linux /dev/ttyACM0, macOS /dev/tty.usbmodem..., Windows COM3 etc.
(LeRobot's `lerobot-find-port` finds it.) On Linux you may need to be in the dialout group.

The servos are read raw, exactly as LeRobot reads them for its calibration. The
calibration you made with LeRobot is stored in the servos (homing offset), so the pose you
held when you pressed ENTER reads 2047 for every joint. The printout shows each joint's
angle from that pose, which is a quick way to check the calibration.
"""

import argparse
import http.client
import json
import math
import sys
import time
from urllib.parse import parse_qs, urlparse

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SHORT = ("pan", "lift", "elbow", "wflex", "roll")  # for the printout

# Feetech STS3215 (protocol 0, little-endian) control table and instructions
ADDR_TORQUE_ENABLE = 40
ADDR_PRESENT_POSITION = 56
INST_READ, INST_WRITE, INST_SYNC_READ = 0x02, 0x03, 0x82
BROADCAST_ID = 0xFE


def packet(servo_id, instruction, params):
    body = [servo_id, len(params) + 2, instruction, *params]
    return bytes([0xFF, 0xFF, *body, (~sum(body)) & 0xFF])


def decode_position(lo, hi):
    value = lo | (hi << 8)
    return -(value & 0x7FFF) if value & 0x8000 else value  # bit 15 is the sign


class FeetechBus:
    def __init__(self, port, baudrate):
        import serial  # pyserial

        self.ser = serial.Serial(port, baudrate, timeout=0.02)
        self.use_sync_read = True

    def _read_exact(self, n):
        data = self.ser.read(n)
        if len(data) != n:
            raise TimeoutError("no answer from servo")
        return data

    def _read_status(self):
        """One status packet -> (id, error byte, params)."""
        prev = 0
        for _ in range(64):  # skip any stray bytes before the 0xFF 0xFF header
            b = self._read_exact(1)[0]
            if prev == 0xFF and b == 0xFF:
                break
            prev = b
        else:
            raise IOError("no packet header")
        servo_id, length = self._read_exact(2)
        rest = self._read_exact(length)  # error, params..., checksum
        if (~(servo_id + length + sum(rest[:-1]))) & 0xFF != rest[-1]:
            raise IOError("bad checksum")
        return servo_id, rest[0], rest[1:-1]

    def write_byte(self, servo_id, address, value):
        self.ser.reset_input_buffer()
        self.ser.write(packet(servo_id, INST_WRITE, [address, value]))
        try:
            self._read_status()
        except (TimeoutError, IOError):
            pass

    def read_positions(self, ids):
        if self.use_sync_read:
            try:
                self.ser.reset_input_buffer()
                self.ser.write(packet(BROADCAST_ID, INST_SYNC_READ, [ADDR_PRESENT_POSITION, 2, *ids]))
                got = {}
                for _ in ids:
                    servo_id, _, params = self._read_status()
                    got[servo_id] = decode_position(params[0], params[1])
                return [got[i] for i in ids]
            except (TimeoutError, IOError, KeyError, IndexError):
                self.use_sync_read = False  # some adapters drop sync-read answers: read one by one
        positions = []
        for servo_id in ids:
            self.ser.reset_input_buffer()
            self.ser.write(packet(servo_id, INST_READ, [ADDR_PRESENT_POSITION, 2]))
            got_id, _, params = self._read_status()
            if got_id != servo_id:
                raise IOError(f"asked servo {servo_id}, got {got_id}")
            positions.append(decode_position(params[0], params[1]))
        return positions


class FakeArm:
    """A slow wave around the calibration pose, to test without hardware."""

    def read_positions(self, ids):
        t = time.time()
        return [2047 + 300 * math.sin(0.5 * t + i) for i in range(len(ids))]


class Poster:
    """Posts to <url>/leader?token=... over one kept-alive connection."""

    def __init__(self, url):
        u = urlparse(url)
        token = parse_qs(u.query).get("token", [None])[0]
        if not token:
            sys.exit("the URL must contain ?token=... (copy it from the teleop output)")
        self.host, self.port = u.hostname or "localhost", u.port or 80
        self.path = "/leader?token=" + token
        self.conn = None

    def post(self, body):
        for attempt in (0, 1):
            try:
                if self.conn is None:
                    self.conn = http.client.HTTPConnection(self.host, self.port, timeout=2)
                self.conn.request("POST", self.path, body, {"Content-Type": "application/json"})
                resp = self.conn.getresponse()
                resp.read()
                if resp.status == 403:
                    sys.exit("server says 403: wrong token. Use the URL of the CURRENT teleop job.")
                return True
            except (OSError, http.client.HTTPException):
                self.conn = None
                if attempt:
                    return False
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url", help='teleop URL, e.g. "http://localhost:22874/?token=XYZ" (quote it)')
    p.add_argument("--port", default="/dev/ttyACM0", help="serial port of the leader arm")
    p.add_argument("--arm", choices=("right", "left"), default="right", help="which sim arm follows the leader")
    p.add_argument("--hz", type=float, default=50.0)
    p.add_argument("--baudrate", type=int, default=1_000_000)
    p.add_argument("--ids", type=int, nargs=6, default=[1, 2, 3, 4, 5, 6], help="servo ids, in joint order")
    p.add_argument("--fake", action="store_true", help="no hardware: send a slow wave")
    args = p.parse_args()

    if args.fake:
        arm = FakeArm()
    else:
        try:
            arm = FeetechBus(args.port, args.baudrate)
        except ImportError:
            sys.exit("pyserial is missing: pip install pyserial")
        except OSError as e:
            sys.exit(f"cannot open {args.port}: {e}")
        for servo_id in args.ids:  # a leader arm is moved by hand: torque off, as LeRobot does
            arm.write_byte(servo_id, ADDR_TORQUE_ENABLE, 0)
    poster = Poster(args.url)
    print(f"leader arm -> sim {args.arm} arm, {args.hz:.0f} Hz. Ctrl+C to stop.")

    period = 1.0 / args.hz
    sent, failed_reads, last_print, server_ok = 0, 0, time.time(), None
    try:
        while True:
            t0 = time.time()
            try:
                ticks = arm.read_positions(args.ids)
                failed_reads = 0
            except (TimeoutError, IOError) as e:
                failed_reads += 1
                if failed_reads in (1, 50):
                    print(f"\nservo read failed ({e}); check power and the USB cable", flush=True)
                time.sleep(period)
                continue  # nothing is sent, so the sim gives the arm back to keys after 0.5 s

            ok = poster.post(json.dumps({args.arm: ticks}))
            if ok != server_ok:
                print("\nconnected to the teleop server" if ok else
                      f"\ncannot reach {args.url.split('?')[0]}: is the teleop job running and the port forwarded?",
                      flush=True)
                server_ok = ok
            sent += ok

            if t0 - last_print >= 0.5:
                rate = sent / (t0 - last_print)
                sent, last_print = 0, t0
                angles = " ".join(f"{j}={(t - 2047) * 360 / 4096:+6.1f}" for j, t in zip(SHORT, ticks))
                print(f"\r{rate:4.0f} Hz | deg from calib pose: {angles} | gripper ticks {ticks[5]:.0f}   ",
                      end="", flush=True)
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
