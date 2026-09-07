# ============================================================
# 11주차 — 연산 그래프 최적화와 런타임
# 추론 런타임이 수행하는 그래프 변환을 직접 구현합니다.
#
# 이 파일의 TODO 를 채워 과제를 완성하세요.
# 함수 시그니처와 출력 형식은 변경하지 마세요.
# 제출: git push (별도의 제출 버튼은 없습니다)
# ============================================================

import sys


def topo_sort(nodes, deps):
    """노드 의존 관계를 위상 정렬해 실행 순서를 반환한다.

    같은 조건이면 노드 ID 가 작은 것을 먼저 실행합니다.
    """
    # TODO: 진입 차수를 세어 위상 정렬하세요.
    raise NotImplementedError


def fold_constants(nodes, deps, values):
    """입력이 모두 상수인 노드를 미리 계산해 상수로 바꾼다.

    ADD 는 입력 상수들의 합으로 접습니다. 접힌 노드는 CONST 가 됩니다.
    반환: 접기 후 남은 노드 수
    """
    # TODO: 상수로 접을 수 있는 노드를 찾아 처리하세요.
    raise NotImplementedError


def fuse_conv_relu(nodes, deps):
    """CONV 바로 뒤에 RELU 가 하나만 연결되어 있으면 두 노드를 하나로 합친다.

    반환: 융합 후 노드 수
    """
    # TODO: CONV -> RELU 패턴을 찾아 합치세요.
    raise NotImplementedError


def eliminate_dead(nodes, deps, output_id):
    """출력 노드에서 역방향으로 도달 가능한 노드만 남긴다.

    반환: 살아남은 노드 ID 의 오름차순 리스트
    """
    # TODO: 출력에서 역추적해 도달 가능 집합을 구하세요.
    raise NotImplementedError


def main():
    lines = [l.split() for l in sys.stdin.read().strip().split("\n")]
    n = int(lines[0][0])
    nodes, deps, values = {}, {}, {}
    for row in lines[1:1 + n]:
        nid, op = int(row[0]), row[1]
        nodes[nid] = op
        if op == "CONST":
            values[nid] = float(row[2]); deps[nid] = []
        else:
            deps[nid] = [int(v) for v in row[2:]]
    out_id = int(lines[1 + n][0])

    print(" ".join(str(v) for v in topo_sort(nodes, deps)))
    print(fold_constants(dict(nodes), dict(deps), dict(values)))
    print(fuse_conv_relu(dict(nodes), dict(deps)))
    print(" ".join(str(v) for v in eliminate_dead(dict(nodes), dict(deps), out_id)))


if __name__ == "__main__":
    main()
