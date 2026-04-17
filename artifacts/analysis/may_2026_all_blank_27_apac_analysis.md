# May 2026 all-blank APAC analysis

Prediction file: `artifacts/predictions/2026_05_strategy_predictions_catboost_3m.csv`

All-blank APAC count: **27**

## Key read
- All 27 are `LOW` risk, so output shows only top-1 prediction per day. If rank 1 is `-`, no backup option is shown.
- These accounts do have history, but blank probability won rank 1 across all 10 day columns.
- Use the CSV for per-day blank probability and best alternative label/probability.

## Reason bucket counts
| reason | count |
| --- | --- |
| Communication exists, but model probability still ranked no-contact highest for this LOW profile. | 27 |

## Compact account table
| APAC_CARD_NUMBER | nonblank_schedule_slots_3m | all_blank_history_month_count | sms_total_intensity_3m | wh_total_intensity_3m | voice_total_intensity_3m | success_signal_3m | failed_signal_3m | avg_blank_probability | min_blank_probability | strongest_alt_day | strongest_alt_label | strongest_alt_probability |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MFLAPDSECUL000005126588 | 12 | 0 | 11 | 11 | 12 | 30 | 4 | 0.738 | 0.4139 | D-5 | SMS-9AM-REGIONAL | 0.4408 |
| MFLKTKSECUL000005066611 | 13 | 1 | 11 | 11 | 12 | 30 | 4 | 0.6927 | 0.5129 | D-5 | SMS-9AM-REGIONAL | 0.4677 |
| MFLKTKSECUL000005322420 | 14 | 0 | 12 | 12 | 11 | 31 | 4 | 0.7235 | 0.5612 | D+1 | SMS-4PM-ENGLISH | 0.4053 |
| MFLKTKSECUL000005349103 | 13 | 1 | 11 | 11 | 11 | 20 | 13 | 0.6638 | 0.5034 | D+4 | SMS-10AM-REGIONAL | 0.4595 |
| MFLMPDSECUL000005324198 | 15 | 0 | 10 | 12 | 10 | 31 | 1 | 0.7384 | 0.4656 | D-5 | SMS-9AM-REGIONAL | 0.4398 |
| MFLRAJSECUL000005151616 | 13 | 0 | 11 | 10 | 10 | 25 | 6 | 0.7025 | 0.5488 | D+2 | SMS-8AM-REGIONAL | 0.4337 |
| MFLSECUGUJL000005675864 | 13 | 0 | 11 | 11 | 9 | 24 | 7 | 0.7012 | 0.5416 | D+2 | SMS-8AM-REGIONAL | 0.4411 |
| MFLSECUKTKL000005377613 | 13 | 1 | 10 | 11 | 11 | 25 | 7 | 0.7017 | 0.4572 | D-3 | WH-9AM-KANNADA | 0.4588 |
| MFLSECUKTKL000005400963 | 15 | 0 | 10 | 12 | 10 | 31 | 1 | 0.731 | 0.4892 | D-5 | SMS-9AM-REGIONAL | 0.3813 |
| MFLSECUKTKL000005424935 | 12 | 0 | 10 | 12 | 11 | 31 | 2 | 0.719 | 0.4519 | D-5 | SMS-9AM-REGIONAL | 0.4426 |
| MFLSECUKTKL000005479001 | 12 | 1 | 11 | 11 | 10 | 21 | 11 | 0.6586 | 0.5022 | D+4 | SMS-10AM-REGIONAL | 0.4595 |
| MFLSECUKTKL000005506962 | 13 | 0 | 12 | 12 | 11 | 30 | 5 | 0.7272 | 0.515 | D-5 | SMS-9AM-REGIONAL | 0.4343 |
| MFLSECUKTKL000005648053 | 14 | 0 | 12 | 11 | 10 | 33 | 0 | 0.7446 | 0.4768 | D-5 | SMS-9AM-REGIONAL | 0.4215 |
| MFLSECUTAML000005425539 | 13 | 0 | 11 | 11 | 12 | 30 | 4 | 0.7112 | 0.4427 | D-5 | SMS-9AM-REGIONAL | 0.4806 |
| MFLSECUTELL000005508338 | 13 | 1 | 11 | 11 | 10 | 21 | 11 | 0.676 | 0.5115 | D+4 | SMS-10AM-REGIONAL | 0.46 |
| MFLSECUWBGL000005633969 | 10 | 0 | 6 | 8 | 5 | 16 | 3 | 0.769 | 0.5294 | D-4 | SMS-9AM-ENGLISH | 0.3872 |
| MFLTAMSECUL000005074754 | 16 | 0 | 11 | 11 | 11 | 32 | 1 | 0.7118 | 0.4373 | D-5 | SMS-9AM-REGIONAL | 0.4826 |
| MFLTAMSECUL000005237852 | 10 | 1 | 9 | 8 | 8 | 19 | 6 | 0.7285 | 0.6021 | D-5 | SMS-9AM-REGIONAL | 0.3691 |
| MFLTELSECUL000005078894 | 14 | 0 | 11 | 11 | 11 | 29 | 4 | 0.7294 | 0.4779 | D-5 | SMS-9AM-REGIONAL | 0.4093 |
| MFLTELSECUL000005139424 | 12 | 0 | 9 | 11 | 9 | 27 | 2 | 0.7757 | 0.5148 | D-5 | SMS-9AM-REGIONAL | 0.4622 |
| MFLTELSECUL000005300557 | 15 | 0 | 10 | 12 | 10 | 28 | 4 | 0.7134 | 0.5069 | D-3 | WH-9AM-TELUGU | 0.4342 |
| MFLUNSEKERL000005749059 | 1 | 0 | 4 | 1 | 1 | 5 | 1 | 0.7423 | 0.4972 | D-4 | SMS-9AM-ENGLISH | 0.4562 |
| MFLUNSEKTKL000005767848 | 13 | 0 | 11 | 12 | 11 | 34 | 0 | 0.7343 | 0.4251 | D+1 | SMS-4PM-ENGLISH | 0.3806 |
| MFLUNSEORIL000005690753 | 14 | 0 | 10 | 12 | 9 | 31 | 0 | 0.7427 | 0.4911 | D-5 | SMS-9AM-REGIONAL | 0.4695 |
| MFLUNSETAML000005727883 | 13 | 0 | 11 | 10 | 11 | 31 | 1 | 0.7504 | 0.4969 | D-5 | SMS-9AM-REGIONAL | 0.435 |
| MFLUNSETAML000005807432 | 2 | 0 | 2 | 0 | 0 | 2 | 0 | 0.8516 | 0.577 | D-4 | SMS-9AM-ENGLISH | 0.4034 |
| MFLUTTSECUL000005280828 | 12 | 1 | 11 | 11 | 12 | 29 | 5 | 0.6829 | 0.4503 | D-5 | SMS-9AM-REGIONAL | 0.4847 |

Full CSV has per-day blank probability and best alternative columns.
