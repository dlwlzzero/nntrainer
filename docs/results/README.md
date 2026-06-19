# 디바이스 테스트 결과 기록

이 폴더(`docs/results/`)는 디바이스 테스트 결과 기록의 루트다. **HTP 백엔드의
결과는 하위 폴더 [`htp_results/`](./htp_results/)에 기록**한다. (다른 백엔드/주제의
결과가 생기면 같은 방식으로 형제 하위 폴더를 추가한다.)

HTP 백엔드를 단말(디바이스)에서 테스트할 때마다 그 결과를 `htp_results/`에 누적
기록한다. 유닛 테스트(Task 2)든 E2E 추론(Task 5)이든, **디바이스에서 테스트를
실행했다면 반드시 `docs/results/htp_results/`에 결과 파일을 추가**한다.

## 기록 규칙 (필수)

1. **위치: `docs/results/htp_results/`.** HTP 결과는 모두 이 하위 폴더에 넣는다.
2. **실행 1회 = 결과 파일 1개.** 기존 파일을 덮어쓰지 말고 새 파일을 추가한다
   (실행 이력이 시간순으로 남아야 함).
3. **날짜와 시간 기재는 필수.** 파일명과 본문 양쪽에 기재한다. 타임존(KST 등)도
   함께 적는다. 실행 시점에 아래 명령으로 타임스탬프를 얻는다:
   ```bash
   date '+%Y-%m-%d %H:%M:%S %Z'      # 본문용,  예: 2026-06-18 14:32:07 KST
   date '+%Y%m%d-%H%M%S'             # 파일명용, 예: 20260618-143207
   ```
4. **파일명 규칙:** `htp_results/<test-kind>-<device>-<YYYYMMDD-HHMMSS>.md`
   - `<test-kind>`: `unittest` | `e2e`
   - `<device>`: 단말 약칭 (예: `s25ultra`)
   - 예) `htp_results/unittest-s25ultra-20260618-143207.md`,
     `htp_results/e2e-s25ultra-20260619-091500.md`

## 본문 템플릿

새 결과 파일은 아래 양식을 채워 작성한다(빈 항목은 `N/A`로 둔다).

```markdown
# HTP <test-kind> 결과 — <device>

- **일시(date/time):** YYYY-MM-DD HH:MM:SS KST   ← 필수
- **단말:** Galaxy S25 Ultra (SM-S938), Snapdragon 8 Elite (SM8750), Hexagon V79, arm64-v8a
- **DSP_ARCH:** v75 (build target; HW는 V79, SDK 6.0.0.2 최대 v75)
- **Hexagon SDK:** 6.0.0.2 (/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.0.0.2)
- **브랜치 / 커밋:** htp_libs_integration @ <git rev-parse --short HEAD>
- **빌드 옵션:** -Denable-htp=true -Dmmap-read=false ...
- **실행자:** <name>

## 결과 요약

| 테스트 | PASS | FAIL | 비고 |
|--------|------|------|------|
| unittest_htp_mat_mul   |  |  |  |
| unittest_htp_quantizer |  |  |  |
| unittest_htp_rms_norm  |  |  |  |

(E2E의 경우: prefill/generation TPS, peak RSS, 호출된 HTP 커널 등을 표로 기재)

## 원본 로그 (발췌)

```
<adb 실행 출력 핵심 부분 붙여넣기>
```

## 판정

- [ ] 전부 PASS → 다음 단계 진행 가능
- 실패 항목 / 후속 조치: ...
```

## 비고

- 이 폴더는 사람이 읽는 기록용이다. 대용량 원본 로그 전체가 아니라 판정에 필요한
  발췌를 남긴다(필요 시 전체 로그 경로를 본문에 명시).
- 디바이스 세대가 다를 수 있으므로(원 검증 S23/V73 ↔ 현재 S25 Ultra/V79) **단말과
  DSP_ARCH를 매 기록에 반드시 명기**한다.
