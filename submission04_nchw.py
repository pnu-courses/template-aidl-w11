"""제4문제 (NCHW) — 전처리 + MobileNetV2 stem: uint8 이미지 -> 정규화 -> conv3x3/2 + BN + ReLU6   [Halide 스케줄]

    preprocess_stem_nchw(img, mean, std, w, bn)
        img  (N, C, H, W)     = (4, 3, 224, 224)   uint8 4 장 (0..255)
        mean, std (3,)              float32  채널별 정규화 상수 (실제 크기는 ImageNet 값, 작은 크기는 다른 값)
        w    OIHW             = (32, 3, 3, 3)      편향 없음, stride 2 · padding 1
        bn   (4, C_out)            = (4, 32)            [gamma, beta, running_mean, running_var], eps 1e-5
        ->  y                      = (4, 32, 112, 112)  float32

    계산 (PyTorch 와 같은 결과 — 7주차 04 와 같은 문제)
        x = (img / 255 - mean) / std                              # 채널별, float32
        a = conv(x, w, stride 2, padding 1)                       # 패딩은 정규화한 x 에 0 을 붙인다
        y = relu6(bn(a))
        H_out = (H + 2 - 3) // 2 + 1                              (W 도 같은 식)

생각할 점 — 출력 (4 x 32 x 112 x 112 = 6.4 MB) 쓰기가 계산보다 무거울 수 있다 (메모리에 막힌 문제 — 강의의 블러).
정규화 · BN 은 일차식이라 가중치 · 편향으로 접을 수 있다 (접는 단계를 작은 Func 로 두고 compute_root).
함정 — 패딩은 정규화한 공간의 0 이다 (원래 이미지 쪽에서 0 을 붙이면 테두리가 틀린다).

Halide 로 쓴다 (11주차 강의: 알고리즘과 스케줄의 분리)
    알고리즘 (_algorithm) 은 무엇을 계산하는지만 적는다. 출발점 코드의 알고리즘은 그대로 맞으니 고칠 필요가 없다.
    스케줄 (_schedule) 이 어떤 순서로 · 어디에 저장하며 계산할지를 정한다. 출발점은 단계마다 compute_root —
    단계마다 전체 중간 결과를 만들고 다음 단계가 읽는다 (7주차의 '융합하지 않은 연산 사슬') — 이고, 벡터도 타일도 없다.
    여기에 split · tile · reorder · vectorize · unroll · compute_at · store_at 을 걸어 빠르게 만든다.
    알고리즘을 바꿔도 된다 (예: BN 을 가중치에 접는 단계를 따로 두기, 가중치를 다시 늘어놓는 Func) — 값만 같으면 된다.

    좌표 순서 주의 — Halide 는 첫 좌표가 가장 안쪽 (메모리에서 연속) 축이다. 이 파일의 L(c, i, j, n) 이
    텐서 좌표를 NCHW 의 Halide 좌표 순서로 바꿔 준다. 그래서 두 레이아웃 파일의 알고리즘은 같은 글자인데,
    메모리 순서가 달라 좋은 스케줄은 서로 다르다 (vectorize 할 축이 어디가 연속인지에 달렸다).

    파이프라인은 처음 한 번만 JIT 컴파일해 (compile_to_callable) 모듈 전역에 두고, 호출마다 새 입력으로 실행한다.
    입력 크기가 바뀌어도 (작은 크기 검사) 다시 컴파일하지 않는다. 그래서 출력 Func 를 split · tile 할 때는
    tail=hl.TailStrategy.GuardWithIf 를 주자 — 기본값 (ShiftInwards) 은 출력이 split 크기보다 작으면 (작은 크기의
    가로 9 칸 같은) 실행 중 오류가 난다. bound() 로 크기를 못 박아도 작은 크기 검사에서 틀린다.
    가중치를 다시 늘어놓는 Func (wf ...) 와 패딩 Func (xp ...) 는 출발점에서 inline 이다 — compute_root 하면 한 번만
    만들어 두고 읽는다 (Func 의 저장 순서는 인자 순서: 첫 인자가 연속 축. reorder_storage 로 바꿀 수 있다).

검증 (uv run verify.py submission04_nchw.py)   ->  success 또는 fail
    규칙    단일 스레드 · Halide 로 계산 — 아래 금지 목록. 어기면 fail
            실행 검사: 실제로 돌려서 스레드가 늘거나 여러 코어를 쓰거나, Halide 파이프라인 안에서 보낸 시간이
            50% 미만이면 fail. 채점기가 HL_NUM_THREADS=1 로 돌리므로 parallel() 은 빨라지지 않는다
    정확도  실제 크기 + 작은 크기 2 개 (다른 H, W, C)에서 float64 기준값과 상대오차 2e-4 이하,
            출력 모양 · float32 · 입력 배열 불변. 시간을 재는 호출의 출력도 매번 검사한다
    시간    Raspberry Pi 5 · 코어 1 개. 준비 3 번 뒤 9 번 재서 (호출마다 다른 입력 · 가중치 · BN)
            가장 빠른 2 번과 가장 느린 2 번을 뺀 5 번의 평균 (ms) -> 리더보드
            참고: 출발점 코드 74.3 ms · 목표 6.5 ms (조교 정답 x 1.25). 7주차 같은 문제의 Numba 기록과 비교해 보자

금지
    numpy 행렬곱·선형대수·FFT: dot · matmul · @ · einsum · tensordot · inner · outer · linalg · fft ·
                              convolve · correlate · x.dot(...)
    Halide 의 바깥 코드 · 대상 바꾸기: define_extern · Internal · load_plugin · Target · compile_to_* (callable 만 허용) ·
                              apply_autoscheduler (자동 스케줄러는 분석 스크립트에서 돌리고 결과 스케줄을 옮겨 적는다)
    numpy · halide · math 밖의 import (numba · scipy · torch · 같은 폴더의 다른 파일 포함)
허용
    Halide 의 모든 알고리즘 · 스케줄 명령, numpy 의 배열 생성·변형·원소별 연산

확인: uv run submission04_nchw.py      (= uv run verify.py submission04_nchw.py)
전체 규칙: uv run verify.py --rules
"""
import numpy as np
import halide as hl

#################### START EDIT IT HERE  ####################

def L(c, i, j, n):
    """텐서 좌표 (채널 c, 가로 i, 세로 j, 이미지 n) -> Halide 좌표. numpy (N, C, H, W) = Halide (W, H, C, N)."""
    return (i, j, c, n)


def LW(co, ci, ky, kx):
    """가중치 좌표 -> Halide 좌표. numpy OIHW (C_out, C_in, KH, KW) = Halide (KW, KH, C_in, C_out)."""
    return (kx, ky, ci, co)


CH, HD, WD = 2, 1, 0        # Halide 차원 번호: 채널 · 세로 · 가로 (x.dim(CH).extent() = C)

def _algorithm():
    img = hl.ImageParam(hl.UInt(8), 4, "img")
    mean = hl.ImageParam(hl.Float(32), 1, "mean")
    std = hl.ImageParam(hl.Float(32), 1, "std")
    w = hl.ImageParam(hl.Float(32), 4, "w")
    bn = hl.ImageParam(hl.Float(32), 2, "bn")
    c, i, j, n = hl.Var("c"), hl.Var("i"), hl.Var("j"), hl.Var("n")
    H, W = img.dim(HD).extent(), img.dim(WD).extent()

    norm = hl.Func("norm")
    norm[L(c, i, j, n)] = (hl.f32(img[L(c, i, j, n)]) / 255.0 - mean[c]) / std[c]
    xp = hl.Func("x_pad")                                    # 정규화한 공간에서 0 패딩
    inside = (i >= 0) & (i < W) & (j >= 0) & (j < H)
    xp[L(c, i, j, n)] = hl.select(inside, norm[L(c, hl.clamp(i, 0, W - 1), hl.clamp(j, 0, H - 1), n)], 0.0)

    co, ci, kx, ky = hl.Var("co"), hl.Var("ci"), hl.Var("kx"), hl.Var("ky")
    wf = hl.Func("wf")                                     # 가중치를 (co, ci, kx, ky) 순서로 — 저장하면 co 가 연속
    wf[co, ci, kx, ky] = w[LW(co, ci, ky, kx)]
    r = hl.RDom([(0, 3), (0, 3), (0, 3)], "r")               # r.x = kx, r.y = ky, r.z = ci
    conv = hl.Func("conv")
    conv[L(c, i, j, n)] = hl.f32(0)
    conv[L(c, i, j, n)] += wf[c, r.z, r.x, r.y] * xp[L(r.z, 2 * i + r.x - 1, 2 * j + r.y - 1, n)]
    y = hl.Func("y")
    y[L(c, i, j, n)] = hl.clamp((conv[L(c, i, j, n)] - bn[c, 2]) * (bn[c, 0] / hl.sqrt(bn[c, 3] + hl.f32(1e-5)))
                                + bn[c, 1], 0.0, 6.0)
    return dict(inputs=[img, mean, std, w, bn], out=y, norm=norm, x_pad=xp, conv=conv, wf=wf, vars=(c, i, j, n), r=r)

def _schedule(s):
    """출발점: 단계마다 compute_root. 벡터 · 타일 없음."""
    s["norm"].compute_root()
    s["conv"].compute_root()

def _compile():
    """알고리즘 + 스케줄 -> 한 번만 JIT 컴파일한 Callable. 호출: _RUN(입력 Buffer ..., 출력 Buffer)."""
    s = _algorithm()
    _schedule(s)
    return s["out"].compile_to_callable(s["inputs"], hl.get_jit_target_from_environment())


_RUN = None

def preprocess_stem_nchw(img, mean, std, w, bn):
    """uint8 -> 정규화 -> conv3x3/2 -> BN -> ReLU6. 반환은 float32 연속 배열 (NCHW)."""
    global _RUN
    if _RUN is None:
        _RUN = _compile()
    n, _, h, wd = img.shape
    co = w.shape[0]
    y = np.empty((n, co, (h - 1) // 2 + 1, (wd - 1) // 2 + 1), np.float32)
    _RUN(hl.Buffer(img), hl.Buffer(mean), hl.Buffer(std), hl.Buffer(w), hl.Buffer(bn), hl.Buffer(y))
    return y

#################### END EDIT IT HERE  ####################

def main():
    import verify
    raise SystemExit(0 if verify.main(["04_nchw"]) else 1)


if __name__ == "__main__":
    main()
