# 전체 모듈 설명서 — **코드에서 생성된다**

> `python competition/tools/build_manual.py` 로 다시 만든다.
> 각 줄의 설명은 그 파일 docstring 의 첫 문장이고, 실행 예시도
> docstring 에서 뽑는다 — **손으로 고치지 말 것.** 설명을 바꾸려면
> 모듈의 docstring 을 고친다.

**규모**: 도메인·모델 모듈 84개 · 빌더 27개 (src 111개) · 대시보드 뷰 25개 · 테스트 103개

## 먼저 읽을 것 셋

1. `docs/CAPABILITIES.md` — 되는 것과 **안 되는 것(§E)**
2. `docs/REPORT_INPUT.md` — 제안서를 쓸 때 인용할 사실 전부
3. `docs/AIHUB_71763.md` · `docs/PREREGISTRATION.md` — 데이터 감사와 사전등록 규약

## 이 저장소의 규율 다섯

1. **계산을 재구현하지 않는다.** 서버도 화면도 모듈을 부른다.
2. **상수를 화면에 박지 않는다.** 정본 모듈에서 주입한다.
3. **문턱을 발명하지 않는다.** 자기 이력의 경보율 대역에서 역산한다.
4. **근거 있는 것만 신고한다.** 못 내는 것은 못 낸다고 답한다.
5. **등급을 섞지 않는다.** 실측/계산/유도/합성/지침.

## 기둥 1 — 발정·교배 관리

농장이 오늘 쓰는 번식 판단.

| 모듈 | 하는 일 | 실행 |
|---|---|---|
| `breeding_ledger` | 개체 발정·임신 통합 관리표 + 향후 관리 일정. | `직접 실행` |
| `breeding_timing` | 교배 적기(AI timing) 산출 + 회전율·경제 효과 — 교배사 핵심 로직. | `직접 실행` |
| `estrus_calendar` | 71471 외음부 라벨 → **개체별 발정 달력**, 그리고 bbox 프레임과의 연결. | `직접 실행` |
| `estrus_early_warning` | 발정 조기경보 — 이유/초교배 기준 일(day) 단위 D-day 예측 + 지연·무발정 경보. | `직접 실행` |
| `estrus_link` | 행동 → 발정 연계 (behavior-to-estrus linkage). | `직접 실행` |
| `estrus_onset` | 시간 윈도우 기반 발정 시작점(onset) 탐지. | `직접 실행` |
| `feeding_monitor` | 개체별 섭취 모니터링 — 합사 스톨의 먹이 경쟁을 CCTV로 계량한다. | `직접 실행` |
| `herd_board` | 모돈군 현황판 — 주차별 임신돈·포유모돈, 산차 구성, 도태·후보돈 전입 계획. | `직접 실행` |
| `mating_plan` | 교배 배정 계획 — 농장장의 표를 그대로, 손 대신 산식으로 채운다. | `직접 실행` |
| `pipeline_gilt` | 통합 파이프라인: CCTV 발정관찰 → 후보돈 무발정 위험 + 개선 처방. | `직접 실행` |
| `pregnancy_check` | 임신진단 3단계 체크포인트 — 재발돈을 언제 잡느냐가 공태일을 결정한다. | `직접 실행` |
| `repro_calendar` | 번식 작업 캘린더 — 날짜 하나로 전체 일정을 자동 생성(입력 간소화). | `직접 실행` |
| `repro_cause_attribution` | 번식 문제 유형 분류 + 원인 귀인(cause attribution). | `직접 실행` |
| `stall_estrus` | 교배사(스톨) 전용 발정 지표 — 활동량이 없는 환경에서 발정을 읽는다. | `직접 실행` |
| `vision_contract` | 영상 모델과 앱 사이의 **계약** — 모델은 갈아끼우는 부품이다. | `직접 실행` |
| `work_log` | 작업 로그 — 무엇을 언제 누가 했는지 쌓고, 그것으로 일정을 정정한다. | `직접 실행` |

<details><summary>실행 예시</summary>

```
python competition/src/breeding_ledger.py
python competition/src/breeding_timing.py        # 시연
python competition/src/estrus_calendar.py <외음부라벨디렉터리> [bbox라벨디렉터리]
python competition/src/estrus_early_warning.py     # 합성 시연
python competition/src/estrus_link.py [frames.csv|라벨디렉터리]
python competition/src/feeding_monitor.py     # 합성 시연
python competition/src/herd_board.py            # 시연(합성 300두)
python competition/src/mating_plan.py                      # 합성 시연
python competition/src/mating_plan.py --sows sows.csv --boars boars.csv         [--max-f 0.0625] [--services 3] [--out plan.csv]
python competition/src/pipeline_gilt.py                 # 합성 시연
python competition/src/pipeline_gilt.py <cctv_라벨디렉터리> <gilt_mgmt.csv>
python competition/src/pregnancy_check.py
python competition/src/repro_calendar.py            # 시연
python competition/src/repro_calendar.py 2026-08-10 # 이유일 지정
python competition/src/repro_cause_attribution.py   # 합성 시연
python competition/src/stall_estrus.py     # 합성 시연
python competition/src/vision_contract.py
python competition/src/work_log.py          # 시연(합성 로그 생성 후 집계)
```

</details>

## 기둥 2 — 돈사 운영·행정

지은 돈사에서 무엇이 병목이고 몇 두를 받는가.

| 모듈 | 하는 일 | 실행 |
|---|---|---|
| `barn_watch` | 배치 전이 감시 — **다음 배치로 넘어갈 때마다** 돈사 현황을 찍고 검사한다. | `직접 실행` |
| `batch_flow` | 돈군흐름(배칭) — 개체 단위 연속 흐름이 아니라 **배치 단위**로 관리한다. | `직접 실행` |
| `farm_registry` | 농장 구조 등록 + 돼지관리표 — 축사동 → 돈방/군사 → 개체의 3계층. | `직접 실행` |
| `farm_scale` | 등록 규모 검산 — **총사육수와 상시모돈은 다른 수다.** | `직접 실행` |
| `growth_flow` | 사육단계 관리 — 번식에서 출하까지. | `직접 실행` |
| `improve_path` | 현재 성적 ↔ **이 농장이 올릴 수 있는 상한**, 그리고 그 사이의 경로. | `직접 실행` |
| `legal_density` | 법정 사육밀도 — 「축산법 시행령」[별표 1] 돼지 표를 **조문 그대로.** | `직접 실행` |
| `perf_formula` | 번식 성적 공식 — **공식 정의를 정본으로, 입력 변수를 이름으로 받는다.** | `직접 실행` |
| `run_farm` | 모돈 두수 하나로 전체를 돌린다 | `직접 실행` |
| `synth_farm` | 가상 농장 데이터 생성 — **실측 분포를 재현하고, 재현했는지 검사한다.** | `직접 실행` |
| `table_export` | 표 내보내기 — 화면에서 본 것을 파일로 가져간다. | `직접 실행` |

<details><summary>실행 예시</summary>

```
python competition/src/barn_watch.py --sows 300
python competition/src/barn_watch.py --setup my_farm.json     # 등록 화면 JSON
python competition/src/batch_flow.py            # 간격별 비교 + 배치 배정
python competition/src/batch_flow.py 21         # 3주 간격으로
python competition/src/farm_registry.py
python competition/src/farm_scale.py        # 합성 시연 (등급 합성)
python competition/src/growth_flow.py
python competition/src/improve_path.py        # 합성 시연 (등급 합성)
python competition/src/legal_density.py      # 표와 출처 출력
python competition/src/perf_formula.py        # 합성 시연 (등급 합성)
python competition/src/run_farm.py --sows 300
python competition/src/run_farm.py --sows 300 --npd 62 --weaned 10
python competition/src/run_farm.py --setup my_farm.json --herd my_herd.csv
python competition/src/run_farm.py --data     # 필요한 자료가 뭔지만 출력
python competition/src/synth_farm.py --sows 300 --years 2
python competition/src/synth_farm.py --sows 300 --csv /tmp/herd.csv
python competition/src/table_export.py --sheet capacity
```

</details>

## 환경·생체 — 지침 층 + 자기 기준선 층

위험(지침 위반)과 주의(평소와 다름)를 겹으로 본다.

| 모듈 | 하는 일 | 실행 |
|---|---|---|
| `barn_env_control` | 돈사 환경 위험 알람 — 지침 대역(절대)과 자기 기준선 편차(상대)를 **겹으로** 본다. | `직접 실행` |
| `barn_environment` | 축사 환경(온습도) → 번식 영향 계산 — ICT 데이터를 숫자 표시로 끝내지 않는다. | `직접 실행` |
| `behavior_baseline` | 행동 기준선 층 — 구성비의 **자기 기준선 편차**로 발정·분만·질병을 본다. | `직접 실행` |
| `behavior_head_train` | 행동 헤드 가중치 학습 — **등가중을 이길 때만** 교체 후보가 된다. | `직접 실행` |
| `behavior_vocab` | 행동 어휘 축소 — **응용이 쓰는 넷으로 접어서 다시 잰다.** | `직접 실행` |
| `bio_baseline_71763` | 71763 생체지표 **평균 0 기준표** — 사양관리용 이상치 문턱. | `직접 실행` |
| `env_anomaly` | 축사 환경(온·습·환기) 이상치 탐지 — 사분위 표시 + 날짜별 추이 + 알림 목록. | `직접 실행` |
| `env_scale` | 돈사 환경 −1~1 편차 스케일 + 지침 위험 표시 — 71763 클립 CSV 를 먹는다. | `직접 실행` |
| `vision_pig_behavior` | 업로드된 행동 분할 모델을 `vision_contract` 에 꽂는 어댑터. | `직접 실행` |

<details><summary>실행 예시</summary>

```
python competition/src/barn_env_control.py                # 합성 시연
python competition/src/barn_env_control.py --log env.csv  # barn,stage,time,temp_c,nh3_ppm
python competition/src/barn_environment.py
python competition/src/behavior_baseline.py     # 합성 시연 (등급 합성)
python competition/src/behavior_head_train.py         --dets-dir data/cctv/dets --labels data/cctv/labels/labels.csv         --per 60 --sec-per-frame 30 --head estrus         --positive 발정 --negative 비발정 --out data/cctv/summary/head_estrus.json
python competition/src/behavior_head_train.py     # 합성 시연 (등급 합성)
python competition/src/behavior_vocab.py        # 병합 전/후 재측정
python competition/src/bio_baseline_71763.py --clips clips.csv
python competition/src/bio_baseline_71763.py <라벨디렉터리>
python competition/src/env_anomaly.py <71763_라벨디렉터리>
python competition/src/env_anomaly.py --clips clips.csv
python competition/src/env_anomaly.py --synthetic     # 배관 검증용(합성)
python competition/src/env_scale.py --clips <출력>/clips_71763.csv
python competition/src/env_scale.py --clips barn_log.csv --key barn
python competition/src/vision_pig_behavior.py            # 접목 상태 점검
python competition/src/vision_pig_behavior.py --ckpt pig_polygon_epoch12.pth 프레임들/
```

</details>

## 근거층 — 국내 실측 분석

466행·179농장에서 나온 격차·계절·하락.

| 모듈 | 하는 일 | 실행 |
|---|---|---|
| `eda` | 탐색적 데이터 분석(EDA). | `직접 실행` |
| `estrus_label_audit` | 71471 발정 라벨 감사 — **피처로 쓰기 전에 누수부터 본다**. | `직접 실행` |
| `farm_economics` | 농장 경영 — 생산비·사료·손익, 그리고 **어느 지렛대가 돈이 되는가**. | `직접 실행` |
| `farm_gap` | 내 농장이 국내 분포에서 **얼마나 멀어져 있나** — 순위가 아니라 거리. | `직접 실행` |
| `farm_monthly` | 월별 번식통계 — 계절성과 임신사고 구성. | `직접 실행` |
| `farm_monthly_panel` | 월별 번식통계 **패널** — 계절 손실을 농장별로 따라간다. | `직접 실행` |
| `farm_panel` | 같은 농장의 **연도별 변화** — 성적은 실제로 얼마나 움직이나. | `직접 실행` |
| `korean_farm_stats` | 국내 466개 농장 번식성적 실측 — 문헌값을 실측 분포로 바꾼다. | `직접 실행` |
| `path_predict` | 작업 로그 → **실제 경로** → **다음 사건 예측**. | `직접 실행` |
| `psy_priority` | PSY 회수 우선순위 — 이미 검증된 진단을 **처방 순서로 배열**한다. | `직접 실행` |

<details><summary>실행 예시</summary>

```
python competition/src/estrus_label_audit.py --bbox <bbox_dir> [--vulva <vulva_dir>]
python competition/src/farm_economics.py
python competition/src/farm_gap.py --npd 62 --weaned 10 --farrowing 74
python competition/src/farm_monthly.py
python competition/src/farm_monthly_panel.py --audit
python competition/src/farm_panel.py --sows 300
python competition/src/korean_farm_stats.py [xlsx경로]
python competition/src/path_predict.py --sows 300 --years 3
python competition/src/psy_priority.py --sows 300
python competition/src/psy_priority.py --setup my_farm.json
```

</details>

## 영상·모델 — 학습과 평가

탐지·자세·행동·추적. 되는 범위를 재는 코드까지.

| 모듈 | 하는 일 | 실행 |
|---|---|---|
| `analyze_kaggle_by_reference` | AI Hub 71471 발정 표준으로 케글 데이터를 분석하는 프로그램. | `직접 실행` |
| `analyze_video` | 영상 분석기 — 영상 파일을 넣으면 결과 리포트(HTML)를 만든다. | `직접 실행` |
| `box_merge` | 분할 박스 병합 — 창살·기둥에 가려 한 마리가 여러 박스로 쪼개지는 문제 보정. | `직접 실행` |
| `estrus_contrast_eval` | 개체 내 대조 발정 검증 — 71471 [Bbox] 서브셋의 한계를 실측으로 규명. | `직접 실행` |
| `finetune_polygon` | 622 폴리곤 → YOLO-seg 파인튜닝. | `직접 실행` |
| `iou_tracker` | 간단한 IoU 기반 다개체 추적기. | `직접 실행` |
| `ml_core` | 학습·평가 공통 규약 — 이 프로젝트가 **비싸게 배운 것**을 코드로 굳힌다. | `직접 실행` |
| `model_71471_estrus` | 71471 발정 탐지(estrus detection) 모델. | `직접 실행` |
| `model_71763` | 71763 양돈 생체 에너지 — 파싱→모델링 실행 스크립트. | `직접 실행` |
| `model_behavior_appearance` | 행동 인식 개선 — 외형(크롭) 피처 추가 + before/after 비교. | `직접 실행` |
| `model_edinburgh_behavior` | Edinburgh 실데이터 — 돼지 행동 인식 모델. | `직접 실행` |
| `model_gilt_anestrus` | 후보돈(replacement gilt) 무발정·발정지연 위험 예측 + 개선요인 진단. | `직접 실행` |
| `motion_tracker` | 카메라 모션 보상 추적 — 흔들리는 영상에서도 개체 ID를 유지한다. | `직접 실행` |
| `polygon_shape_eval` | 폴리곤 실루엣이 bbox 보다 | `직접 실행` |
| `pose_vs_behavior_eval` | 자세(keypoints) vs 행동라벨(bbox) — 같은 과제·같은 설계로 정보량 비교. | `직접 실행` |
| `posture_crop_feats` | 자세 크롭 외형 피처 — bbox 기하가 카메라를 넘지 못하는 지점을 메운다. | `직접 실행` |
| `posture_crossview` | 교차-뷰 자세 인식 — 병목 해부와 개선. | `직접 실행` |
| `posture_eval` | 교차 데이터셋 자세 검증 도구. | `직접 실행` |
| `posture_features` | 자세 인식 피처 강화 — 교배사(스톨) 발정 판정의 기반. | `직접 실행` |
| `temporal_features` | 시간 윈도우 피처 — 행동 인식에 '최근 맥락'을 넣는다. | `직접 실행` |
| `tracking_sweep` | 추적 파라미터 탐색 — 과분할의 진짜 원인을 찾아 최적 설정을 고른다. | `직접 실행` |
| `train` | 베이스라인 모델링. | `직접 실행` |
| `train_behavior_seq` | 행동 인식 시퀀스 모델 — 프레임 하나가 아니라 **구간**을 본다. | `직접 실행` |
| `train_posture_cnn` | 자세 CNN — **원리적 상한 0.861 을 넘을 수 있는 유일한 경로**를 시험한다. | `직접 실행` |
| `train_pseudo_label` | 자동 라벨링(pseudo-label) 재학습 — 국내 축사 영상으로 탐지기 도메인 적응. | `직접 실행` |
| `validate_estrus_reference` | 발정 실측 검증 — 71471 발정 정답으로 EstrusReference 를 보정·평가. | `직접 실행` |
| `view_align` | 멀티뷰 자세 인식 — 뷰 정합(view alignment) 전/후 정확도 비교. | `직접 실행` |

<details><summary>실행 예시</summary>

```
python competition/src/analyze_kaggle_by_reference.py [71471_frames.csv]
python competition/src/analyze_video.py <영상.mp4> [출력이름]
python competition/src/box_merge.py <영상.mp4>    # 병합 전후 마릿수 비교
python competition/src/estrus_contrast_eval.py <vulva_dir> <bbox_dir>
python competition/src/finetune_polygon.py prep <라벨디렉터리> <이미지디렉터리>
python competition/src/finetune_polygon.py train <라벨디렉터리> <이미지디렉터리>         --max-images 5000 --imgsz 416 --epochs 50
python competition/src/ml_core.py     # 지금 학습 가능한 과제 점검
python competition/src/model_71471_estrus.py competition/data/aihub/71471
python competition/src/model_71763.py <라벨디렉터리>       # 실데이터
python competition/src/model_71763.py --clips clips.csv    # 이미 접어 둔 표
python competition/src/model_71763.py                      # 합성 시연
python competition/src/model_behavior_appearance.py
python competition/src/model_edinburgh_behavior.py [frames.csv 또는 라벨디렉터리]
python competition/src/motion_tracker.py <영상.mp4>   # 보정 전후 비교
python competition/src/polygon_shape_eval.py <라벨디렉터리>
python competition/src/pose_vs_behavior_eval.py <keypoints_dir>
python competition/src/posture_crop_feats.py        # 추출 + 캐시
python competition/src/posture_crossview.py            # 전체 비교
python competition/src/posture_crossview.py --quick    # 폴드 3개만
python competition/src/posture_features.py [view|image|naive]
python competition/src/temporal_features.py [frames.csv]   # 유무 비교 평가
python competition/src/tracking_sweep.py <영상.mp4>
python competition/src/train.py            # 두 과제 모두
python competition/src/train.py reg        # 회귀만
python competition/src/train.py clf        # 분류만
python competition/src/train_behavior_seq.py --quick
python competition/src/train_behavior_seq.py --epochs 25
python competition/src/train_posture_cnn.py --cache      # 크롭 추출만
python competition/src/train_posture_cnn.py --quick      # 1폴드 속도 측정
python competition/src/train_posture_cnn.py              # 전체 LOVO
python competition/src/train_pseudo_label.py build <영상들...> --out /tmp/kr_pig
python competition/src/train_pseudo_label.py train /tmp/kr_pig --epochs 8
python competition/src/validate_estrus_reference.py [라벨디렉터리]
python competition/src/view_align.py [대회경로]
```

</details>

## 데이터 — 파서·수집

원자료를 표로 바꾸는 층. 여기서 감사도 한다.

| 모듈 | 하는 일 | 실행 |
|---|---|---|
| `aihub` | AI Hub(aihub.or.kr) 데이터 API 클라이언트. | `직접 실행` |
| `aihub_bridge` | AI Hub 71471 실데이터 → 운영 모듈 연결. | `직접 실행` |
| `aihub_estrus_reference` | AI Hub 71471(돼지 발정행동) 기준(reference) 모듈. | 라이브러리 |
| `fetch_622` | 622 폴리곤 학습 세트 받기 — 국내망에서 한 번에. | `직접 실행` |
| `generate_data` | 양돈 스마트팜 합성 데이터 생성기. | `직접 실행` |
| `parse_71471_keypoints` | AI Hub 71471 [Keypoints] 서브셋 파서 + 자세(pose) 피처. | `직접 실행` |
| `parse_71471_real` | AI Hub 71471 **실제 라벨 스키마** 파서 (labelon 배포본). | `직접 실행` |
| `parse_71763_batch` | 71763 라벨 36만개 → 프레임·클립 CSV **한 번만** 만드는 배치 파서. | `직접 실행` |
| `parse_aihub` | AI Hub 라벨 → 분석용 정형 테이블(train.csv) 파서. | `직접 실행` |
| `parse_edinburgh` | Edinburgh Pig Behaviour Annotated 데이터셋 파서. | `직접 실행` |
| `parse_pig_polygon` | Pig_Polygon(분만행위 폴리곤) 데이터셋 수집 경로. | `직접 실행` |

<details><summary>실행 예시</summary>

```
python competition/src/aihub.py search 양돈
python competition/src/aihub.py list
python competition/src/aihub.py tree 71408
python competition/src/aihub.py download 71408 509489,509492   # 라벨링데이터만
python competition/src/aihub_bridge.py            # 연동 + 능력 매트릭스
python competition/src/fetch_622.py            # 라벨만 (85MB)
python competition/src/fetch_622.py --images   # 라벨 + TS06 (10GB)
python competition/src/parse_71471_keypoints.py <라벨디렉터리>
python competition/src/parse_71471_real.py <라벨디렉터리>
python competition/src/parse_71763_batch.py <라벨디렉터리> [--out 디렉터리]
python competition/src/parse_71763_batch.py <라벨디렉터리> --clean
python competition/src/bio_baseline_71763.py --clips <출력>/clips_71763.csv
python competition/src/env_anomaly.py --clips <출력>/clips_71763.csv
python competition/src/parse_pig_polygon.py            # 자체 점검(합성)
python competition/src/parse_pig_polygon.py <디렉터리>   # 실데이터 파싱
```

</details>

## 화면 — 대시보드 빌더

정적 HTML 을 만든다. 계산은 위 모듈을 부른다.

| 모듈 | 하는 일 | 실행 |
|---|---|---|
| `build_activity_analysis` | 탐지 기반 파생 행동분석 — 축사·세션별 점유 heatmap + 마릿수·활동량 추이. | `직접 실행` |
| `build_alert_console` | 실시간 경보 콘솔 — 관리자 아침 점검용 우선순위 큐 + 모바일 알림 목업. | `직접 실행` |
| `build_analysis_report` | 케글 돼지 데이터 종합 분석 리포트 (HTML). | `직접 실행` |
| `build_app_prototype` | 동작하는 앱 프로토타입 — 눌러서 돌아다니는 단일 HTML. | `직접 실행` |
| `build_app_screens` | 앱 사용 화면 — 실제로 농가가 보게 될 화면을 실데이터로 채워 보인다. | `직접 실행` |
| `build_barn_map` | 농장 도면 기반 실시간 관제 — 축사 배치도 위에 관리대상돈을 띄운다. | `직접 실행` |
| `build_behavior_gallery` | 행동 확인 프론트 — 여러 농장 CCTV 피드(장척 영상) + 프레임 갤러리. | `직접 실행` |
| `build_breeding_console` | 번식 관리 콘솔 — 캘린더·현황판·임신진단·관리표를 한 화면에. | `직접 실행` |
| `build_dashboard` | 대시보드 데이터 생성 → competition/dashboard/index.html. | `직접 실행` |
| `build_dashboard_hub` | 통합 대시보드 허브 — 6개 웹 뷰 + 리포트를 한 랜딩에서 네비게이션. | `직접 실행` |
| `build_detection_viewer` | 돼지 탐지 뷰어 프론트 — 파일명/축사(pen)/소스로 검색 → 탐지 즉시 표시. | `직접 실행` |
| `build_edinburgh_dashboard` | Edinburgh 실데이터 → 활동 모니터링 대시보드(HTML). | `직접 실행` |
| `build_env_anomaly` | 환경 이상치 화면 — 온·습·환기 사분위 밴드 + 날짜별 추이 + 알림 목록. | `직접 실행` |
| `build_estrus_timeline` | 개체 추적 → 개체별 발정 타임라인 (실영상). | `직접 실행` |
| `build_eval_report` | 평가 리포트 — 혼동행렬·PR·ROC·보정곡선 (심사 신뢰도). | `직접 실행` |
| `build_farm_diagnosis` | 실측 진단 대시보드 — 466농장 분포·격차·패널을 화면에 올린다. | `직접 실행` |
| `build_farm_setup` | 농장 최초 등록 — 규모·축사동·운영 방식을 받아 **바로 진단으로 넘긴다**. | `직접 실행` |
| `build_kaggle_notebooks` | 캐글 노트북 생성 — GPU 에서 돌릴 학습 노트북 두 개를 만든다. | `직접 실행` |
| `build_ops_console` | 교배 배정 · 환경 알람 — **서버 없이 열리는** 판. | `직접 실행` |
| `build_pc_console` | PC 관리 콘솔 — 사무실에서 쓰는 화면. | `직접 실행` |
| `build_pc_suite` | PC 통합 콘솔 — 사무실에서 쓰는 화면 여섯을 **한 파일**로 합친다. | `직접 실행` |
| `build_pigflow_console` | 돈군흐름 관제 — 분만틀에서 역산한 설계, 그리고 어디서 막히는가. | `직접 실행` |
| `build_posture_gallery` | 자세 인식 프론트 생성 — 사진 갤러리 + 주석 영상. | `직접 실행` |
| `build_posture_report` | 자세 인식 병목 리포트 — 무엇이 병목이었고 무엇을 고쳤는가. | `직접 실행` |
| `build_reference_report` | AI Hub 71471 발정 표준 기반 케글 분석 — 웹 리포트(HTML). | `직접 실행` |
| `build_repro_dashboard` | 번식 문제 진단 · 발정 조기경보 대시보드 (자체완결 HTML, 연결 불필요). | `직접 실행` |
| `build_season_interval` | 여름 손실 · 간격 what-if — **서버 없이 열리는** 판. | `직접 실행` |

<details><summary>실행 예시</summary>

```
python competition/src/build_activity_analysis.py [데이터셋경로]
python competition/src/build_alert_console.py
python competition/src/build_analysis_report.py
python competition/src/build_app_prototype.py
python competition/src/build_app_screens.py
python competition/src/build_barn_map.py
python competition/src/build_behavior_gallery.py            # 캐시 영상 자동탐색
python competition/src/build_behavior_gallery.py <folder> <color.mp4>  # 단일 지정
python competition/src/build_breeding_console.py
python competition/src/build_dashboard.py                       # 합성 시연
python competition/src/build_dashboard.py <cctv_dir> <mgmt.csv> # 실데이터
python competition/src/build_dashboard_hub.py
python competition/src/build_detection_viewer.py [데이터셋경로]
python competition/src/build_edinburgh_dashboard.py [frames.csv|라벨디렉터리]
python competition/src/build_env_anomaly.py
python competition/src/build_estrus_timeline.py [녹화폴더] [color.mp4]
python competition/src/build_eval_report.py
python competition/src/build_farm_diagnosis.py
python competition/src/build_farm_setup.py
python competition/src/build_kaggle_notebooks.py
python competition/src/build_ops_console.py
python competition/src/build_pc_console.py
python competition/src/build_pc_suite.py
python competition/src/build_pigflow_console.py
python competition/src/build_posture_gallery.py [대회경로]
python competition/src/build_posture_report.py
python competition/src/build_reference_report.py [71471_frames.csv]
python competition/src/build_repro_dashboard.py
python competition/src/build_season_interval.py
```

</details>

## 도구 (`tools/`)

| 도구 | 하는 일 |
|---|---|
| `baseline_from_dets` | CLI 검출 JSONL → 시간창 → 행동 기준선 — **로컬 실증용 브리지.** |
| `build_breeding_board` | 농장 화이트보드 교배표 → HTML 대시보드. |
| `build_manual` | 전체 파이썬 모듈 설명서를 **코드에서 생성**한다 |
| `check_docs` | 문서에 박힌 숫자가 실제와 맞는지 검사한다. |
| `check_download` | 내려받은 파일이 왜 tar 가 아닌지 진단한다. |
| `merge_check` | 합류 점검 — 다른 자리에서 온 브랜치를 **합치기 전에** 본다. |

## 검증

```
python competition/tests/smoke_test.py     # 전체 테스트
python competition/tools/check_docs.py     # 문서 수치 대조
```
