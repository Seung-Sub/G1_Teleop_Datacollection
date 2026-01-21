# control/grav_lut.py
# -*- coding: utf-8 -*-
import numpy as np

# G1 인덱스(질문 코드 기준)
WAIST_YAW, WAIST_ROLL, WAIST_PITCH = 12, 13, 14
KNEE_L, KNEE_R = 3, 9

class GravLUT:
    """
    허리(pitch/roll) 2D bilinear, 무릎 1D linear 보간으로
    중력피드포워드 토크를 근사해 줍니다.
    """
    def __init__(self, npz_path: str):
        z = np.load(npz_path)
        self.wp = z["wp"]            # waist pitch grid
        self.wr = z["wr"]            # waist roll  grid
        self.k  = z["k"]             # knee grid
        self.lut_waist = z["lut_waist"]  # (len(wp), len(wr), 2)  [τ_pitch, τ_roll]
        self.lut_knee  = z["lut_knee"]   # (len(k),)

    @staticmethod
    def _interp1(x, xp, fp):
        i = np.searchsorted(xp, x, side="left")
        i = int(np.clip(i, 1, len(xp) - 1))
        x0, x1 = xp[i-1], xp[i]
        y0, y1 = fp[i-1], fp[i]
        t = (x - x0) / ( (x1 - x0) + 1e-12 )
        return (1 - t) * y0 + t * y1

    def _interp2(self, x, y, xg, yg, F):
        # bilinear (x→wp, y→wr)
        i = int(np.clip(np.searchsorted(xg, x, side="left"), 1, len(xg) - 1))
        j = int(np.clip(np.searchsorted(yg, y, side="left"), 1, len(yg) - 1))
        x0, x1 = xg[i-1], xg[i]
        y0, y1 = yg[j-1], yg[j]
        f00 = F[i-1, j-1]
        f01 = F[i-1, j  ]
        f10 = F[i  , j-1]
        f11 = F[i  , j  ]
        tx = (x - x0) / ( (x1 - x0) + 1e-12 )
        ty = (y - y0) / ( (y1 - y0) + 1e-12 )
        return (1-tx)*(1-ty)*f00 + (1-tx)*ty*f01 + tx*(1-ty)*f10 + tx*ty*f11  # shape (2,)

    def ff_from_q(self, q_23: np.ndarray):
        """
        q_23: (23,) 현재 관절각 [rad]
        return:
          leg_tau_ff   (12,)
          waist_tau_ff (3,)
        """
        wp = float(q_23[WAIST_PITCH])
        wr = float(q_23[WAIST_ROLL])
        kL = float(q_23[KNEE_L])
        kR = float(q_23[KNEE_R])

        tau_wp_wr = self._interp2(wp, wr, self.wp, self.wr, self.lut_waist)  # [τ_pitch, τ_roll]
        tau_kL    = self._interp1(kL, self.k, self.lut_knee)
        tau_kR    = self._interp1(kR, self.k, self.lut_knee)

        leg_tau_ff   = np.zeros(12, dtype=np.float32)
        waist_tau_ff = np.zeros(3,  dtype=np.float32)

        # 허리: [yaw, roll, pitch]
        waist_tau_ff[1] = float(tau_wp_wr[1])  # roll(13)
        waist_tau_ff[2] = float(tau_wp_wr[0])  # pitch(14)

        # 무릎 좌/우
        leg_tau_ff[ KNEE_L ] = float(tau_kL)   # index 3
        leg_tau_ff[ KNEE_R ] = float(tau_kR)   # index 9

        return leg_tau_ff, waist_tau_ff
