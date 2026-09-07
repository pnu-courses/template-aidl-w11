# 11주차 과제 — 연산 그래프 최적화와 런타임

> **Ch 9. 효율적인 컴퓨터 자원 활용** · 배점 20점 · 마감 2026-11-16 23:59

## 학습 목표

추론 런타임이 수행하는 그래프 최적화를 직접 구현합니다. 상수 접기, 연산자 융합, 불필요 노드 제거를 순서대로 적용합니다.

## 다루는 개념

- 연산 그래프의 노드와 간선 표현
- 상수 접기(constant folding)
- 연산자 융합(operator fusion)
- 죽은 노드 제거(dead code elimination)
- 위상 정렬과 실행 순서
- ONNX Runtime 과 TVM 이 하는 일

## 구현할 내용

각 파일의 `TODO` 를 채우세요. 함수 시그니처와 `main.py` 의 출력 형식은 바꾸지 마세요.

1. `topo_sort(nodes, edges)` — 위상 정렬로 실행 순서 결정
2. `fold_constants(graph)` — 입력이 모두 상수인 노드를 미리 계산
3. `fuse_conv_relu(graph)` — Conv 뒤에 ReLU 가 오면 하나로 합침
4. `eliminate_dead(graph, outputs)` — 출력에 기여하지 않는 노드 제거

## 입출력 형식

**입력**

```
첫 줄: N  (노드 수)
다음 N 줄: 노드ID 연산종류 입력ID들...  (상수는 CONST 값)
마지막 줄: 출력 노드 ID
```

**출력**

```
1줄: 위상 정렬 순서
2줄: 상수 접기 후 남은 노드 수
3줄: 융합 후 노드 수
4줄: 죽은 노드 제거 후 최종 노드 ID 목록 (오름차순)
```

## 예시

입력

```
5
0 CONST 2
1 CONST 3
2 ADD 0 1
3 CONV 2
4 RELU 3
4
```

출력

```
0 1 2 3 4
3
2
2 3
```

## 제출 방법

1. `main.py` 를 완성합니다.
2. `git add . && git commit -m "solve" && git push` — **push 가 곧 제출입니다.**
3. 저장소의 **Actions** 탭에서 자동 채점 결과를 확인합니다.
4. 마감 전까지 몇 번이든 다시 제출할 수 있으며, 마지막 제출이 평가됩니다.

## 평가 기준

| 항목 | 배점 |
|---|---|
| 그래프 최적화 전체 | 10 |
| 죽은 노드 제거 | 5 |
| 융합 없음 | 5 |
| **합계** | **20** |

## 더 읽을거리

- [MIT 6.5940 Lecture 11: TinyEngine and Parallel Processing](https://efficientml.ai)

---

인공지능융합설계및실험 · 2026-fall · 담당 권용인
