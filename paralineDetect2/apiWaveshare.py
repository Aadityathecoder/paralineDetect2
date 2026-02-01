uvicorn apiWaveshare:app --host 0.0.0.0 --port 8000
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from threading import Lock
from pathlib import Path
import json, time

from PCA9685 import PCA9685

pwm = PCA9685(0x40, debug=False)
pwm.setPWMFreq(50)

Dir = ['forward', 'backward']

class MotorDriver:
    def __init__(self):
        self.PWMA = 0
        self.AIN1 = 1
        self.AIN2 = 2
        self.PWMB = 5
        self.BIN1 = 3
        self.BIN2 = 4

    def MotorRun(self, motor: int, index: str, speed: int):
        speed = max(0, min(100, int(speed)))
        if motor == 0:
            pwm.setDutycycle(self.PWMA, speed)
            if index == Dir[0]:
                pwm.setLevel(self.AIN1, 0); pwm.setLevel(self.AIN2, 1)
            else:
                pwm.setLevel(self.AIN1, 1); pwm.setLevel(self.AIN2, 0)
        else:
            pwm.setDutycycle(self.PWMB, speed)
            if index == Dir[0]:
                pwm.setLevel(self.BIN1, 0); pwm.setLevel(self.BIN2, 1)
            else:
                pwm.setLevel(self.BIN1, 1); pwm.setLevel(self.BIN2, 0)

    def MotorStop(self, motor: int):
        if motor == 0:
            pwm.setDutycycle(self.PWMA, 0)
        else:
            pwm.setDutycycle(self.PWMB, 0)

    def Tank(self, left: float, right: float):
        def side(motor, val):
            if abs(val) < 1e-3:
                self.MotorStop(motor)
                return
            sp = int(abs(val) * 100)
            idx = 'forward' if val > 0 else 'backward'
            self.MotorRun(motor, idx, sp)

        side(0, left)   # left motor
        side(1, right)  # right motor

MOTOR = MotorDriver()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

STATE_PATH = Path(__file__).with_name("robotState.json")
stateLock = Lock()

class ControlData(BaseModel):
    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False
    command: str | None = None
    speed: float = Field(0.6, ge=0.0, le=1.0)
    duration: float = Field(0.6, ge=0.05, le=5.0)

robotState = {
    "up": False, "down": False, "left": False, "right": False,
    "command": "stop", "command_id": 0, "timestamp": int(time.time()),
    "speed": 0.6, "duration": 0.6,
}

def write_state_to_disk(state: dict):
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(STATE_PATH)

def cmd_to_tank(cmd: str, sp: float) -> tuple[float,float]:
    c = (cmd or "").lower()
    if c in ("forward", "start", "move"): return sp, sp
    if c in ("back", "backward"):         return -sp, -sp
    if c == "left":                        return -sp, sp
    if c == "right":                       return sp, -sp
    return 0.0, 0.0

def drive_for(left: float, right: float, seconds: float):
    MOTOR.Tank(left, right)
    time.sleep(seconds)
    MOTOR.Tank(0.0, 0.0)

@app.get("/")
def root():
    return {"ok": True, "driver": "Waveshare PCA9685 + MotorDriver", "pwm_freq": 50}

@app.get("/control/status")
def status():
    with stateLock:
        return robotState

@app.post("/control/stop")
def stop():
    MOTOR.Tank(0.0, 0.0)
    with stateLock:
        robotState.update({"command": "stop"})
        robotState["command_id"] += 1
        robotState["timestamp"] = int(time.time())
        write_state_to_disk(robotState)
    return {"message": "stopped", "state": robotState}

@app.post("/control/set")
def update_controls(data: ControlData, bg: BackgroundTasks):
    sp = max(0.0, min(1.0, float(data.speed)))
    dur = float(data.duration)

    cmd = data.command
    if not cmd:
        if   data.up:    cmd = "forward"
        elif data.down:  cmd = "back"
        elif data.left:  cmd = "left"
        elif data.right: cmd = "right"
        else:            cmd = "stop"

    L, R = cmd_to_tank(cmd, sp)

    if (L != 0 or R != 0) and dur > 0:
        bg.add_task(drive_for, L, R, dur)
    else:
        MOTOR.Tank(L, R)

    with stateLock:
        robotState.update({
            "up": data.up, "down": data.down, "left": data.left, "right": data.right,
            "command": cmd, "speed": sp, "duration": dur
        })
        robotState["command_id"] += 1
        robotState["timestamp"] = int(time.time())
        write_state_to_disk(robotState)

    return {"message": "Updated & driving", "state": robotState}
