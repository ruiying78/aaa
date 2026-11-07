# 국장 AAA v6 자동 리밸런싱

## A. 내 PC에서 보기 (더블클릭)
1. Python 설치 (https://www.python.org/downloads — 설치 시 "Add Python to PATH" 체크)
2. `config.json`에 투자금(capital_krw)과 현재 보유수량(holdings) 입력 — 처음 한 번만
3. Windows: `run_AAA.bat` 더블클릭 / Mac: `run_AAA.command` 더블클릭
   → 최신 가격을 받아 계산하고 결과 화면이 브라우저로 열립니다 (첫 실행은 패키지 설치로 1~2분)

엑셀도 갱신하려면: `python aaa_rebalance.py --excel 내전략.xlsx`

## B. GitHub로 자동 갱신 페이지 만들기 (휴대폰에서 링크로 보기)
1. https://github.com 가입 → 우측 상단 + → New repository
   - 이름: `aaa` (아무거나) / **Public** 선택 / Create
2. "uploading an existing file" 클릭 → 이 폴더의 파일을 모두 끌어다 놓기
   - **config.json은 올리지 마세요** (투자금·보유수량이 공개됨)
   - `.github` 폴더가 안 보이면: Add file → Create new file → 이름에
     `.github/workflows/update.yml` 입력 후 update.yml 내용 붙여넣기
3. Settings → Pages → Source: "Deploy from a branch" → Branch: `main` / 폴더 `/docs` → Save
4. Actions 탭 → "AAA 리밸런싱 자동 갱신" → Run workflow (최초 1회 수동 실행)
5. 1~2분 뒤 `https://<내아이디>.github.io/aaa/` 접속 → 휴대폰 홈화면에 추가

이후 평일 매일 아침(베이징 06:17) 자동 갱신됩니다. 바로 갱신하고 싶으면 4번처럼 Run workflow.
공개 페이지에는 비중만 표시되고, 금액·보유수량은 PC 버전에서만 나옵니다.
