"""Two-link arm. MuJoCo when it imports, otherwise a kinematic hold/track step. The log records which one ran."""

from __future__ import annotations

import numpy as np

XML = """
<mujoco>
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <body name="base">
      <joint name="j1" type="hinge" axis="0 0 1" range="-2 2" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.02"/>
      <body pos="0.3 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" range="-2.5 2.5" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.018"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a1" joint="j1" kp="20" ctrlrange="-2 2"/>
    <position name="a2" joint="j2" kp="20" ctrlrange="-2.5 2.5"/>
  </actuator>
</mujoco>
"""


def integrate(q: np.ndarray, target: np.ndarray | None, gain: float = 0.25) -> np.ndarray:
    if target is None:
        return q.copy()
    return q + gain * (np.asarray(target, dtype=np.float64) - q)


class ArmSim:
    timestep = 0.01  # seconds per step, matches <option timestep> above

    def __init__(self, backend: str = "auto"):
        self.q = np.zeros(2, dtype=np.float64)
        self.backend = "kinematic"
        self._model = None
        self._data = None
        self._mujoco = None
        if backend == "kinematic":
            return
        try:
            import mujoco

            self._mujoco = mujoco
            self._model = mujoco.MjModel.from_xml_string(XML)
            self._data = mujoco.MjData(self._model)
            self.backend = "mujoco"
        except Exception:
            if backend == "mujoco":
                raise
            self.backend = "kinematic"

    def step(self, target: list[float] | None, n: int = 8) -> np.ndarray:
        if self.backend == "mujoco":
            if target is None:
                # Drop residual velocity. Hold means the joint position stays.
                self._data.qvel[:] = 0
                self._data.ctrl[:] = self._data.qpos[:2]
                self._mujoco.mj_forward(self._model, self._data)
                self.q = self._data.qpos[:2].copy()
                return self.q.copy()
            self._data.ctrl[:] = np.asarray(target, dtype=np.float64)
            for _ in range(n):
                self._mujoco.mj_step(self._model, self._data)
            self.q = self._data.qpos[:2].copy()
            return self.q.copy()
        for _ in range(n):
            self.q = integrate(self.q, None if target is None else np.asarray(target, dtype=np.float64), gain=0.25)
        return self.q.copy()
