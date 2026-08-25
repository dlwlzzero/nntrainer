# M1: RPC 골격 — 인터페이스와 메모리

[← 개요](00-overview.md)

목표: idl+stub/skel 빌드(hexagon-clang 크로스 빌드 + meson 연동), 더미 왕복 성공.

## RPC 인터페이스

**IDL (`nntr_htp.idl`)** — QAIC 컴파일러가 stub(호스트)/skel(DSP) 마샬링 코드를 생성:

```
interface nntr_htp {
  open(domain) → handle              // 세션 생성 (cDSP)
  init(oplist_buf, weight_fd, kv_fd, act_fd, io_fd) → status   // 1회
  forward(n_tokens, pos) → status    // 인자는 스칼라뿐
  close(handle)
}
```

핵심 규칙: **forward의 인자는 스칼라 2개.** token_ids/logits는 사전 매핑된 IO 버퍼로 오간다. 큰 버퍼는 init에서 dma-buf fd로 1회 전달해 DSP 주소 공간에 영구 매핑. 이때 DSP가 `버퍼 id → DSP 주소` 테이블(버퍼 오프셋 테이블)을 구성하고, 이후 모든 텐서 참조는 (버퍼 id, 오프셋)으로 해석된다.

## 버퍼 구성 (전부 rpcmem, 초기화 때 확보)

| 버퍼 | 크기 (qwen3-0.6b, w8) | 접근 패턴 |
|---|---|---|
| WEIGHTS | ~0.75 GB (int8 + scale + rope 테이블) | 호스트 쓰기 1회 → DSP 읽기 전용 |
| KV | 4K seq 기준 ~0.2 GB (fp16) | DSP 전용 |
| ACT | 최대 청크 M 기준 수 MB | DSP 전용 |
| IO | token_ids(in) + logits(out) ~0.6 MB | 청크마다 호스트↔DSP |
| OPLIST | 수 KB | 호스트 쓰기 1회 |

**캐시 일관성:** CPU/DSP 캐시가 분리되어 있으므로 flush/invalidate가 필요하다. IO 버퍼는 forward의 정식 in/out 인자로 선언해 FastRPC 드라이버의 자동 처리에 맡긴다. WEIGHTS/OPLIST는 init 시 1회 flush. KV/ACT는 DSP 전용이라 해당 없음.

## 세션 설정 (open 시 1회)

- cDSP 도메인, unsigned PD (HVX는 특권 불필요. 개발 장비는 testsig 필요할 수 있음)
- `HAP_power` 투표: DCVS·클럭 상한 요청 (누락 시 저클럭으로 동작)
- 워커 스레드 풀: 워커 수 = `qurt_hvx_get_units()` 런타임 조회값 (칩마다 4~6, 하드코딩 금지)

**버전 핸드셰이크:** op-list 포맷 version 필드를 init에서 skel이 검증 — 호스트 라이브러리와 skel .so 불일치 배포 사고 방지.

## 에러 처리

| 상황 | 대응 |
|---|---|
| open/init 실패 (디바이스 없음, testsig, 버전 불일치) | 전체 CPU fallback + warn 로그 |
| init 시 op-list 검증 실패 (경계, 정렬, 미지 op) | DSP가 실행 전 전체 검증 → 에러 반환, 호스트 fallback |
| forward 실패 / DSP 크래시 | fallback 없이 에러 전파 (KV 상태 오염) |
| 가시성 | DSP `FARF` 로깅(logcat 수집), 호스트는 기존 nntrainer 로거 |

원칙: **검증은 전부 init에, forward 경로에는 검사 없음.**
