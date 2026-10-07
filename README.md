# 동물 사진 중복·유사 사진 검출

동일한 사진과 크기만 바뀐 사진을 찾는 Python 도구입니다. 같은 동물의 다른
자세·방향·촬영 각도, 좌우 반전, 회전, 잘린 사진은 별도 사진으로 취급합니다.
사진을 삭제하거나 이동하지 않고 결과만 CSV로 저장합니다.

서로 다른 촬영 사진이라도 같은 동물·비슷한 포즈·비슷한 배경이면 겹치는
후보로 검토하고 싶을 때는 아래 **유사 사진 검사**를 사용하세요.

## train과 test·val 사이의 겹침 검사

`find_split_overlap.py`는 **train × (test + val)**만 비교합니다. 세 폴더는
각각 분리해서 유지하고, `test`나 `val`을 `train` 안에 넣지 마세요.
같은 세트 내부의 유사 사진은 이 보고서에 포함하지 않습니다. train 쪽 사진만
제거 검토 대상으로 표시하고 test·val 사진은 비교 기준으로 유지합니다.

기존 유사 사진 검사의 `.venv`와 설치된 모델을 그대로 사용합니다. 최신
`find_split_overlap.py`, `find_similar_images.py`, `find_duplicate_images.py`가
같은 코드 폴더에 있어야 합니다. CMD 실행 예:

```cmd
.venv\Scripts\python find_split_overlap.py --train "C:\Users\yoonh\OneDrive\바탕 화면\train" --test "C:\Users\yoonh\OneDrive\바탕 화면\test" --val "C:\Users\yoonh\OneDrive\바탕 화면\val" --threshold 0.85 -o split_overlap_results
start "" "split_overlap_results\index.html"
```

`--test`, `--val` 중 하나만 지정해서 검사할 수도 있습니다. 같은 폴더나
서로 포함하는 폴더를 세트로 지정하면 오류로 중단합니다. 결과 폴더는 입력
사진 폴더 밖에 두세요. 폴더 밖의 사진을 가리키는 심볼릭 링크도 거부합니다.

- `index.html`: 왼쪽 **train — 제거 검토 대상**, 오른쪽 **test/val — 유지**.
  정확한 일치인지 유사 후보인지도 표시합니다.
- `split_overlaps.csv`: train 사진, 비교 사진, `test`/`val`, 유사도와 판정 근거.
- `train_candidates.csv`: 제거를 검토할 train 사진의 중복 없는 목록.
  한 train 사진이 여러 비교 사진과 겹칠 수 있으므로 후보 쌍 수와 train 수는 다릅니다.

기본값은 최소 유사도 0.90이며, 임계값 이상인 **모든 세트 간 쌍**을 기록합니다.
`--top-k 3`처럼 명시적으로 지정하면 train당 유사 후보 수가 제한됩니다.
파일·픽셀이 정확히 같은 `exact_file`/`exact_pixels`는 유사도와 후보 수 제한에
관계없이 모두 포함합니다. `near_duplicate_candidate`는 같은 품종이나 배경만
비슷해도 나올 수 있으므로 평가 데이터 누수의 확정 판정으로 사용하지 마세요.

사진은 자동 삭제하지 않습니다. 보고서에서 두 사진을 확인한 뒤 제거하기로
판단한 **train 파일만** 처리하세요. 재검사는 `-o split_overlap_results2`처럼
새 결과 폴더를 지정합니다. 손상된 파일이 있으면 읽은 사진의 보고서를 만들더라도
종료 코드 2와 오류를 표시하므로 전체 검사 성공으로 간주하지 마세요.

## 유사 사진 검사 (Windows CMD)

`find_similar_images.py`는 사전학습된 ResNet50 이미지 특징을 비교해 비슷한 사진
쌍을 점수순으로 찾습니다. 전체 화면과 중앙 정사각형 영역을 따로 비교하고,
전체 유사도 40% + 중앙 유사도 60%로 후보를 정렬합니다. 이미지당 최대 5개의
이웃 후보를 양방향으로 찾으며, 후보들을 자동으로 하나의 중복 그룹으로 합치지
않습니다. 기본 최소 유사도는 0.90입니다.

코드 폴더의 CMD에서, 기존 `.venv`를 사용하세요. 없으면 먼저 생성합니다.

```cmd
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python -m pip check
.venv\Scripts\python find_similar_images.py "C:\Users\yoonh\OneDrive\바탕 화면\train" --threshold 0.90 -o similar_results
start "" "similar_results\index.html"
```

처음 실행할 때 공식 PyTorch 서버에서 약 98MB의 모델을 다운로드합니다.
다운로드의 SHA256 검증을 유지하며, 저장된 모델은 다음 실행에서 재사용합니다.
CPU만으로 실행할 수 있고 GPU·비밀키가 필요하지 않습니다. 모델에는 사진을
업로드하지 않습니다. 입력 사진은 이 PC에서 처리합니다.

- `similar_results/index.html`: 두 사진을 나란히 확인하는 보고서. 파일명 `613`
  등으로 검색하거나 표시할 최소 유사도를 높일 수 있습니다.
- `similar_results/similar_photos.csv`: 후보 파일 경로, 전체·중앙 유사도와 종합 점수.
- 결과 폴더가 이미 있으면 덮어쓰지 않습니다. 재실행은 `-o similar_results2`처럼
  새 폴더를 지정하세요.

후보가 부족하면 `--threshold 0.85`로 낮춰 다시 검사하세요. 더 많은 이웃을
보려면 `--top-k 10`을 추가할 수 있습니다. HTML에서 기준을 낮춰도 최초 검사에서
제외된 후보가 추가되지는 않으므로 이 경우에는 명령을 다시 실행해야 합니다.

이 도구는 **유사 사진 검토 후보**를 찾습니다. 같은 품종의 다른 개체나 다른
포즈도 높은 점수를 받을 수 있고, 배경이 같은 것만으로 점수가 높아질 수도
있습니다. 반대로 비슷한 사진이 기준 아래로 떨어질 수도 있습니다. 모델 점수는
동일한 개체·포즈를 보장하는 확률이 아니므로 최종 판단은 두 사진을 보고 하세요.
CSV의 `match_kind`는 `near_duplicate_candidate`이며 원본 사진은 변경하지 않습니다.

macOS/Linux에서는 가상환경 활성화 후 같은 Python 명령을 사용합니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cpu
python find_similar_images.py /사진/폴더 --threshold 0.90 -o similar_results
```

## 정확한 중복 검사

Python 3.11 이상이 필요합니다. 클라우드 환경은 Python 3.12로 검증했습니다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python find_duplicate_images.py /사진/폴더 -o duplicates.csv
```

폴더의 하위 폴더도 검사합니다. JPG, PNG, WEBP, BMP, TIFF 정지 이미지를
지원합니다. 애니메이션·여러 페이지 이미지는 오류로 보고합니다.
CSV가 이미 있으면 덮어쓰지 않으므로 재실행할 때 새 파일 이름을 지정하세요.

이 클라우드 환경에는 의존성이 이미 설치되어 있습니다.

```bash
cd /workspace/cv1-teamproject
source /workspace/.venvs/cv1-teamproject/bin/activate
python find_duplicate_images.py /사진/폴더 -o duplicates.csv
```

## 결과 해석

`keep`는 유지할 사진, `duplicate`는 중복 사진 또는 검토할 후보입니다.
해상도가 가장 큰 사진을 유지 대상으로 고릅니다. 같은 해상도이면 파일 크기와
경로로 순서를 정합니다. 파일 크기는 화질을 보장하지 않으므로 실제 삭제 전
유지할 사진도 확인하세요.

| match_kind | 의미 |
| --- | --- |
| `exact_file` | 파일 내용이 완전히 동일 |
| `exact_pixels` | 표시되는 RGB 픽셀이 완전히 동일, 메타데이터나 저장 형식은 다를 수 있음 |
| `exact_resize` | Pillow의 지원 리사이즈 방식으로 한 사진에서 다른 사진의 픽셀을 정확히 재현 |
| `resize_candidate` | 압축·다른 리사이즈 방식으로 픽셀이 조금 달라졌지만 두 크기에서 픽셀과 구조가 매우 유사; 직접 확인 필요 |

pHash는 비교 후보를 추리는 데만 사용합니다. 후보는 종횡비, 같은 좌표의 RGB
픽셀, 국소 SSIM을 추가 비교합니다. 이미지 정렬, 회전·반전·자르기 검색은 하지
않습니다. EXIF 방향은 일반 이미지 뷰어처럼 반영하고, 투명 배경은 흰색 위에
표시한 모습으로 비교합니다. 색상 프로파일에 따른 별도 색 변환은 하지 않습니다.

크기 변경이나 JPEG 압축은 정보를 잃으므로 모든 경우에서 원본이 같다는 것을
수학적으로 보장할 수 없습니다. 서로 아주 비슷한 연속 촬영 사진도 추정 후보가
될 수 있어 `resize_candidate`는 **자동 삭제 대상으로 사용하지 마세요**.
보수적인 기준 때문에 압축이 강한 사진, 잘린 사진, 세부 정보가 부족한 작은
사진은 놓칠 수 있습니다. 가로 또는 세로가 64px 미만이면 파일·픽셀 동일성만
검사합니다. 정확한 리사이즈 재현은 대상 크기 800만 픽셀까지 검사합니다.
세밀한 질감을 NEAREST 등으로 강하게 축소해 pHash가 크게 달라진 사진도
후보 추출 단계에서 누락될 수 있습니다.

## 팀원에게 웹 링크로 보고서 공유하기

첨부받은 실제 결과를 `docs/share/`에 웹 공유용으로 준비했습니다.
train 내부 보고서는 9쌍, train과 test·val 사이 보고서는 18쌍입니다.
사진 썸네일과 유사도·파일명·세트 구분은 유지하고 개인 PC의 전체 경로는 제거했습니다.
이 파일들은 업로드한 결과의 스냅샷이며, 검사를 다시 실행해도 자동 갱신되지 않습니다.

저장소의 [GitHub Pages 설정](https://github.com/yoonho1202/cv1-teamproject/settings/pages)에서
**Source → Deploy from a branch**, **Branch → main**, **폴더 → /docs**를 선택하고
**Save**를 누르면 준비한 보고서를 웹에 게시할 수 있습니다.
이미 다른 사이트를 게시 중이면 기존 Pages 설정을 확인한 뒤 변경하세요.
배포가 완료된 뒤 아래 주소가 열리는지 확인하고 공유하세요.

- 전체 보고서: https://yoonho1202.github.io/cv1-teamproject/share/
- train 내부: https://yoonho1202.github.io/cv1-teamproject/share/train.html
- train 대 test·val: https://yoonho1202.github.io/cv1-teamproject/share/split.html

`file:///D:/.../index.html` 주소는 해당 PC의 파일 위치이므로 팀원의 PC에서는
열리지 않습니다. 위 웹 보고서는 설치 없이 브라우저로 볼 수 있으며,
검색과 최소 유사도 필터를 사용할 수 있습니다. 공개된 보고서의 사진은
링크를 가진 사람이 볼 수 있습니다.

## 검증

```bash
python -m unittest discover -s tests -v
```

사용자의 원본 730장과 613/639번 파일은 이 실행 환경에 없습니다. 기본 테스트는
생성한 이미지로 정확한 중복 검사와 유사도 후보 선택·보고서 출력을 검증합니다.
유사 사진 도구는 별도의 공개 동물 사진과 크롭·밝기 변경·무관한 이미지로
실제 사전학습 모델 실행과 CSV/HTML 생성을 검증했습니다. 사용자 데이터에서
적절한 유사도 기준과 개체·포즈 구분은 직접 확인해야 합니다.
