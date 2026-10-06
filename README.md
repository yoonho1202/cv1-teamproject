# 동물 사진 중복 검출

동일한 사진과 크기만 바뀐 사진을 찾는 Python 도구입니다. 같은 동물의 다른
자세·방향·촬영 각도, 좌우 반전, 회전, 잘린 사진은 별도 사진으로 취급합니다.
사진을 삭제하거나 이동하지 않고 결과만 CSV로 저장합니다.

## 실행

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

## 검증

```bash
python -m unittest discover -s tests -v
```

원본 사진 720장은 아직 이 저장소에 없습니다. 테스트는 생성한 이미지로
파일 복사, 서로 다른 저장 형식, 확대·축소, JPEG 재압축 및 다른 사진을 검증합니다.
