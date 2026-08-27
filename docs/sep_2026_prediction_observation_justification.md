# September 2026 Model Prediction Observations - Justification

## Purpose

This document explains why the model generated the observed pre-due recommendations for September 2026, especially LOW-risk cases where some days show no campaign (`-`) and where channels such as Voice Bot or IVR appear more often than SMS.

## Files and Data Used

- Prediction file: `artifacts/predictions/2026_09_strategy_predictions_catboost_3m.csv`
- Raw communication table checked: `digital_collections.communications_aiml_rbl`
- Scope reviewed: `D-5` to `D-1`, risks `LOW`, `MEDIUM`, `HIGH`

## Important Interpretation Rules

- `-` means the model selected no campaign as the top recommendation for that APAC and day.
- LOW-risk output keeps only the top recommendation. If the top recommendation is `-`, alternate recommendations are not shown in the prediction CSV.
- MEDIUM and HIGH risk can include multiple pipe-separated recommendations, so mapping rows can be higher than prediction rows.
- `SMS - language bifurcation is not capturing` usually means SMS recommendations are concentrated in a small number of language labels, mainly `REGIONAL` and `ENGLISH`, because those are the language values learned from the source data/features.

## Prediction Count Summary

| Risk | Day | Total Cases | Blank / No Campaign | Campaign Top-1 | Top-1 Channel Counts | SMS Language Counts | Voice Bot Entries |
|---|---:|---:|---:|---:|---|---|---:|
| LOW | D-5 | 573 | 182 | 391 | `{'SMS': 261, 'VOICE': 129, 'VOICE_BOT': 1}` | `{'REGIONAL': 261}` | 1 |
| LOW | D-4 | 573 | 85 | 488 | `{'VOICE_BOT': 443, 'VOICE': 45}` | `{}` | 443 |
| LOW | D-3 | 573 | 133 | 440 | `{'VOICE': 268, 'VOICE_BOT': 172}` | `{}` | 172 |
| LOW | D-2 | 573 | 206 | 367 | `{'VOICE_BOT': 229, 'VOICE': 136, 'SMS': 2}` | `{'REGIONAL': 2}` | 229 |
| LOW | D-1 | 573 | 55 | 518 | `{'VOICE_BOT': 313, 'VOICE': 171, 'SMS': 34}` | `{'REGIONAL': 34}` | 313 |
| MEDIUM | D-5 | 1751 | 0 | 1751 | `{'SMS': 906, 'VOICE': 843, 'VOICE_BOT': 2}` | `{'REGIONAL': 1553, 'ENGLISH': 86}` | 22 |
| MEDIUM | D-4 | 1751 | 0 | 1751 | `{'VOICE_BOT': 1161, 'VOICE': 439, 'SMS': 151}` | `{'ENGLISH': 496}` | 1842 |
| MEDIUM | D-3 | 1751 | 0 | 1751 | `{'VOICE_BOT': 1158, 'VOICE': 584, 'SMS': 9}` | `{'REGIONAL': 53, 'ENGLISH': 6}` | 1683 |
| MEDIUM | D-2 | 1751 | 0 | 1751 | `{'VOICE_BOT': 938, 'VOICE': 795, 'SMS': 18}` | `{'REGIONAL': 58, 'ENGLISH': 7}` | 1862 |
| MEDIUM | D-1 | 1751 | 0 | 1751 | `{'VOICE_BOT': 1218, 'VOICE': 511, 'SMS': 22}` | `{'REGIONAL': 90, 'ENGLISH': 47}` | 1814 |
| HIGH | D-5 | 6924 | 0 | 6924 | `{'SMS': 5702, 'VOICE': 1209, 'VOICE_BOT': 13}` | `{'REGIONAL': 7045, 'ENGLISH': 320}` | 845 |
| HIGH | D-4 | 6924 | 0 | 6924 | `{'VOICE_BOT': 6052, 'VOICE': 868, 'SMS': 4}` | `{'ENGLISH': 566, 'REGIONAL': 5}` | 8553 |
| HIGH | D-3 | 6924 | 0 | 6924 | `{'VOICE_BOT': 5447, 'VOICE': 1206, 'SMS': 271}` | `{'REGIONAL': 856, 'ENGLISH': 63}` | 10606 |
| HIGH | D-2 | 6924 | 0 | 6924 | `{'VOICE': 3769, 'VOICE_BOT': 2437, 'SMS': 718}` | `{'REGIONAL': 1459, 'ENGLISH': 177}` | 5748 |
| HIGH | D-1 | 6924 | 0 | 6924 | `{'VOICE': 3066, 'VOICE_BOT': 3012, 'SMS': 846}` | `{'ENGLISH': 2085, 'REGIONAL': 30}` | 7139 |

## Why LOW Risk Has Blank Recommendations

For LOW risk, the model returns only the strongest option. If the highest-scoring option is `-`, the final output is blank/no campaign even when the second-best option is SMS, IVR, or Voice Bot.

The no-campaign output does not necessarily mean source data is missing. It means the historical feature pattern looked closer to accounts where no D-day campaign was selected during training.

### Training Target Evidence

The model learned `-` as a valid class from historical training labels. Sample training distribution by risk/day:

- LOW D-5: total training rows `22449`, `-` rows in top labels `3207` (`14.29%`). Top labels: SMS-11AM-ENGLISH: 16005, -: 3207, IVR-6PM-HINDI: 2293, SMS-3PM-REGIONAL: 195, IVR-4PM-HINDI: 189
- LOW D-4: total training rows `22449`, `-` rows in top labels `1110` (`4.94%`). Top labels: SMS-5PM-ENGLISH: 9507, IVR-11AM-HINDI: 6315, SMS-4PM-ENGLISH: 1528, -: 1110, SMS-3PM-ENGLISH: 871
- LOW D-3: total training rows `22449`, `-` rows in top labels `885` (`3.94%`). Top labels: SMS-10AM-ENGLISH: 9347, IVR-2PM-HINDI: 5334, VOICE_BOT-9AM-HINDI: 2157, IVR-9AM-ENGLISH: 1303, VOICE_BOT-9AM-ENGLISH: 955
- LOW D-2: total training rows `22449`, `-` rows in top labels `903` (`4.02%`). Top labels: SMS-10AM-ENGLISH: 8900, IVR-9AM-ENGLISH: 3275, IVR-2PM-HINDI: 2445, IVR-5PM-ENGLISH: 1187, SMS-1PM-ENGLISH: 1086
- LOW D-1: total training rows `22449`, `-` rows in top labels `935` (`4.16%`). Top labels: SMS-4PM-REGIONAL: 8979, IVR-9AM-ENGLISH: 5057, SMS-1PM-ENGLISH: 1865, -: 935, VOICE_BOT-10AM-HINDI: 576
- MEDIUM D-5: total training rows `18301`, `-` rows in top labels `2072` (`11.32%`). Top labels: SMS-11AM-ENGLISH: 13214, -: 2072, IVR-6PM-HINDI: 1568, SMS-12PM-REGIONAL: 567, IVR-4PM-HINDI: 278
- MEDIUM D-4: total training rows `18301`, `-` rows in top labels `1019` (`5.57%`). Top labels: SMS-3PM-ENGLISH: 6523, IVR-5PM-HINDI: 4242, IVR-2PM-HINDI: 1205, SMS-4PM-ENGLISH: 1173, -: 1019
- MEDIUM D-3: total training rows `18301`, `-` rows in top labels `814` (`4.45%`). Top labels: SMS-10AM-ENGLISH: 7790, IVR-2PM-HINDI: 4647, -: 814, IVR-9AM-ENGLISH: 759, VOICE_BOT-9AM-ENGLISH: 606
- MEDIUM D-2: total training rows `18301`, `-` rows in top labels `793` (`4.33%`). Top labels: SMS-10AM-ENGLISH: 7821, IVR-2PM-HINDI: 2591, IVR-9AM-ENGLISH: 2339, -: 793, IVR-5PM-ENGLISH: 498
- MEDIUM D-1: total training rows `18301`, `-` rows in top labels `923` (`5.04%`). Top labels: SMS-1PM-REGIONAL: 5802, IVR-9AM-ENGLISH: 2181, IVR-5PM-HINDI: 1899, SMS-10AM-ENGLISH: 1427, -: 923
- HIGH D-5: total training rows `20629`, `-` rows in top labels `3379` (`16.38%`). Top labels: SMS-11AM-ENGLISH: 6771, SMS-12PM-REGIONAL: 3842, -: 3379, VOICE_BOT-10AM-ENGLISH: 1217, IVR-6PM-HINDI: 1185
- HIGH D-4: total training rows `20629`, `-` rows in top labels `2935` (`14.23%`). Top labels: SMS-3PM-ENGLISH: 3119, -: 2935, IVR-5PM-HINDI: 1890, SMS-1PM-ENGLISH: 1669, SMS-12PM-ENGLISH: 1515
- HIGH D-3: total training rows `20629`, `-` rows in top labels `2874` (`13.93%`). Top labels: SMS-10AM-ENGLISH: 4240, -: 2874, IVR-2PM-HINDI: 2141, VOICE_BOT-10AM-ENGLISH: 1806, SMS-9AM-ENGLISH: 1263
- HIGH D-2: total training rows `20629`, `-` rows in top labels `2803` (`13.59%`). Top labels: SMS-10AM-ENGLISH: 4261, -: 2803, SMS-5PM-REGIONAL: 2197, IVR-2PM-HINDI: 1220, SMS-12PM-ENGLISH: 1015
- HIGH D-1: total training rows `20629`, `-` rows in top labels `2837` (`13.75%`). Top labels: SMS-2PM-REGIONAL: 3076, -: 2837, SMS-5PM-ENGLISH: 2103, SMS-9AM-REGIONAL: 2072, IVR-12PM-HINDI: 1817

## Why SMS Language Bifurcation Looks Weak

SMS language split depends on the normalized `verbiage_language` available in the communications table. In the September output, most SMS recommendations are concentrated under `REGIONAL` and `ENGLISH`. If business expects Hindi/Tamil/etc. for SMS, the source table must contain those language labels for SMS at sufficient volume; otherwise the model cannot learn them as separate SMS strategies.

In short: the model is not inventing a language split. It can only learn SMS language bifurcation from historical SMS rows where language is populated consistently.

## Why Voice Bot Appears in Some Cases

Voice Bot is included as a valid channel. When an APAC has repeated `VOICE_BOT` activity with successful statuses such as `CONNECTED`, the model can rank Voice Bot above SMS/IVR for that day.

For example, APAC `XXXX100X5124` has a D-5 prediction of `VOICE_BOT-10AM-ENGLISH`. Its model input contained recent Voice Bot success at 10AM English, so the D-5 model ranked Voice Bot highest for that APAC.

## Sample Prediction Proof

### LOW D-5 - blank

| Loan Number | Risk | Vertical | EMI Date | Source Month | Prediction Month | Recommendation |
|---|---|---|---|---|---|---|
| XXXX060X1836 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | - |
| XXXX060X5639 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | - |
| XXXX060X6588 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | - |

### LOW D-5 - voice_bot

| Loan Number | Risk | Vertical | EMI Date | Source Month | Prediction Month | Recommendation |
|---|---|---|---|---|---|---|
| XXXX100X5124 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | VOICE_BOT-10AM-ENGLISH |

### LOW D-4 - campaign

| Loan Number | Risk | Vertical | EMI Date | Source Month | Prediction Month | Recommendation |
|---|---|---|---|---|---|---|
| XXXX079X4952 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | VOICE_BOT-10AM-ENGLISH |
| XXXX105X3293 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | VOICE_BOT-10AM-HINDI |
| XXXX055X5731 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | VOICE_BOT-10AM-HINDI |

### LOW D-3 - blank

| Loan Number | Risk | Vertical | EMI Date | Source Month | Prediction Month | Recommendation |
|---|---|---|---|---|---|---|
| XXXX055X1180 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | - |
| XXXX060X1836 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | - |
| XXXX060X3961 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | - |

### LOW D-2 - sms

| Loan Number | Risk | Vertical | EMI Date | Source Month | Prediction Month | Recommendation |
|---|---|---|---|---|---|---|
| XXXX083X6553 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | SMS-12PM-REGIONAL |
| XXXX112X0450 | LOW | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | SMS-12PM-REGIONAL |

### HIGH D-5 - voice_bot

| Loan Number | Risk | Vertical | EMI Date | Source Month | Prediction Month | Recommendation |
|---|---|---|---|---|---|---|
| XXXX050X8731 | HIGH | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | -|SMS-12PM-REGIONAL|VOICE_BOT-10AM-HINDI |
| XXXX055X0135 | HIGH | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | -|SMS-12PM-REGIONAL|VOICE_BOT-10AM-HINDI |
| XXXX055X0378 | HIGH | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | -|SMS-12PM-REGIONAL|VOICE_BOT-10AM-HINDI |

### MEDIUM D-5 - voice_bot

| Loan Number | Risk | Vertical | EMI Date | Source Month | Prediction Month | Recommendation |
|---|---|---|---|---|---|---|
| XXXX077X4294 | MEDIUM | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | SMS-12PM-REGIONAL|VOICE_BOT-10AM-ENGLISH |
| XXXX078X6439 | MEDIUM | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | SMS-12PM-REGIONAL|VOICE_BOT-10AM-ENGLISH |
| XXXX080X4532 | MEDIUM | TWO WHEELER LOAN | 05/09/2026 | AUG-2026 | SEP-2026 | SMS-12PM-REGIONAL|VOICE_BOT-10AM-ENGLISH |

## Raw Communication Evidence for Sample APACs

Below are sample raw pre-due communication counts from `communications_aiml_rbl`. These are not the full model explanation, but they show the historical channel/status/language signals available to the model.

| APAC | EMI Month | Day | Channel | Status | Language | Hour | Count |
|---|---|---|---|---|---|---:|---:|
| XXXX050X8731 | AUG-2026 | D-1 | SMS | UNDELIVERED | English | 12 | 1 |
| XXXX050X8731 | AUG-2026 | D-1 | SMS | UNDELIVERED | Regional | 9 | 1 |
| XXXX050X8731 | AUG-2026 | D-1 | VOICE | OTHER | Hindi | 9 | 1 |
| XXXX050X8731 | AUG-2026 | D-1 | VOICE_BOT | NO_ANSWER | Hindi | 10 | 1 |
| XXXX050X8731 | AUG-2026 | D-1 | VOICE_BOT | NO_ANSWER | Hindi | 14 | 1 |
| XXXX050X8731 | AUG-2026 | D-2 | SMS | UNDELIVERED | English | 12 | 1 |
| XXXX050X8731 | AUG-2026 | D-2 | SMS | UNDELIVERED | Regional | 9 | 1 |
| XXXX050X8731 | AUG-2026 | D-2 | VOICE | NO_ANSWER | Hindi | 9 | 1 |
| XXXX050X8731 | AUG-2026 | D-2 | VOICE | OTHER | English | 14 | 1 |
| XXXX050X8731 | AUG-2026 | D-2 | VOICE_BOT | NO_ANSWER | Hindi | 10 | 1 |
| XXXX050X8731 | AUG-2026 | D-3 | SMS | UNDELIVERED | Regional | 16 | 1 |
| XXXX050X8731 | AUG-2026 | D-3 | VOICE | NO_ANSWER | English | 17 | 1 |
| XXXX050X8731 | AUG-2026 | D-3 | VOICE | NO_ANSWER | Hindi | 9 | 1 |
| XXXX050X8731 | AUG-2026 | D-3 | VOICE_BOT | NO_ANSWER | Hindi | 10 | 1 |
| XXXX050X8731 | AUG-2026 | D-3 | VOICE_BOT | NO_ANSWER | Hindi | 14 | 1 |
| XXXX050X8731 | AUG-2026 | D-4 | SMS | UNDELIVERED | English | 12 | 1 |
| XXXX050X8731 | AUG-2026 | D-4 | SMS | UNDELIVERED | Regional | 16 | 1 |
| XXXX050X8731 | AUG-2026 | D-4 | VOICE | NO_ANSWER | English | 17 | 1 |
| XXXX050X8731 | AUG-2026 | D-4 | VOICE | NO_ANSWER | Hindi | 15 | 1 |
| XXXX050X8731 | AUG-2026 | D-4 | VOICE_BOT | DEQUEUED | Hindi | 13 | 1 |
| XXXX050X8731 | AUG-2026 | D-4 | VOICE_BOT | NO_ANSWER | Hindi | 10 | 1 |
| XXXX050X8731 | AUG-2026 | D-5 | SMS | UNDELIVERED | English | 15 | 1 |
| XXXX050X8731 | AUG-2026 | D-5 | SMS | UNDELIVERED | Regional | 15 | 1 |
| XXXX050X8731 | AUG-2026 | D-5 | VOICE | NO_ANSWER | English | 15 | 1 |
| XXXX050X8731 | AUG-2026 | D-5 | VOICE | NO_ANSWER | Hindi | 16 | 1 |
| XXXX050X8731 | AUG-2026 | D-5 | VOICE_BOT | NO_ANSWER | Hindi | 15 | 1 |
| XXXX050X8731 | FEB-2026 | D-1 | SMS | DELIVERED | English | 15 | 1 |
| XXXX050X8731 | FEB-2026 | D-1 | SMS | UNDELIVERED | English | 10 | 1 |
| XXXX050X8731 | FEB-2026 | D-1 | SMS | UNDELIVERED | English | 17 | 1 |
| XXXX050X8731 | FEB-2026 | D-1 | VOICE | NOT_CONNECTED | English | 9 | 1 |
| XXXX050X8731 | FEB-2026 | D-1 | VOICE | NOT_CONNECTED | Hindi | 11 | 1 |
| XXXX050X8731 | FEB-2026 | D-1 | VOICE | OTHER | English | 17 | 1 |
| XXXX050X8731 | FEB-2026 | D-1 | VOICE | OTHER | Hindi | 14 | 1 |
| XXXX050X8731 | FEB-2026 | D-2 | SMS | DELIVERED | English | 10 | 1 |
| XXXX050X8731 | FEB-2026 | D-2 | SMS | UNDELIVERED | English | 15 | 1 |
| XXXX050X8731 | FEB-2026 | D-2 | SMS | UNDELIVERED | English | 17 | 1 |
| XXXX050X8731 | FEB-2026 | D-2 | VOICE | NOT_CONNECTED | English | 9 | 1 |
| XXXX050X8731 | FEB-2026 | D-2 | VOICE | NOT_CONNECTED | English | 17 | 1 |
| XXXX050X8731 | FEB-2026 | D-2 | VOICE | NOT_CONNECTED | Hindi | 14 | 1 |
| XXXX050X8731 | FEB-2026 | D-2 | VOICE_BOT | NO_ANSWER | Hindi | 11 | 1 |
| XXXX050X8731 | FEB-2026 | D-3 | SMS | DELIVERED | English | 10 | 1 |
| XXXX050X8731 | FEB-2026 | D-3 | SMS | UNDELIVERED | English | 15 | 1 |
| XXXX050X8731 | FEB-2026 | D-3 | SMS | UNDELIVERED | English | 17 | 1 |
| XXXX050X8731 | FEB-2026 | D-3 | VOICE | NOT_CONNECTED | English | 9 | 1 |
| XXXX050X8731 | FEB-2026 | D-3 | VOICE | NOT_CONNECTED | English | 17 | 1 |
| XXXX050X8731 | FEB-2026 | D-3 | VOICE | NOT_CONNECTED | Hindi | 14 | 1 |
| XXXX050X8731 | FEB-2026 | D-3 | VOICE_BOT | NO_ANSWER | Hindi | 11 | 1 |
| XXXX050X8731 | FEB-2026 | D-4 | SMS | DELIVERED | English | 16 | 1 |
| XXXX050X8731 | FEB-2026 | D-4 | VOICE | NOT_CONNECTED | Hindi | 14 | 1 |
| XXXX050X8731 | FEB-2026 | D-4 | VOICE | NOT_CONNECTED | Hindi | 17 | 1 |
| XXXX050X8731 | FEB-2026 | D-5 | SMS | DELIVERED | English | 11 | 1 |
| XXXX050X8731 | JUL-2026 | D-1 | SMS | UNDELIVERED | English | 17 | 1 |
| XXXX050X8731 | JUL-2026 | D-1 | SMS | UNDELIVERED | Regional | 9 | 1 |
| XXXX050X8731 | JUL-2026 | D-1 | VOICE | NO_ANSWER | Hindi | 12 | 1 |
| XXXX050X8731 | JUL-2026 | D-1 | VOICE_BOT | NO_ANSWER | Hindi | 11 | 1 |
| XXXX050X8731 | JUL-2026 | D-1 | VOICE_BOT | NO_ANSWER | Hindi | 15 | 1 |
| XXXX050X8731 | JUL-2026 | D-2 | SMS | UNDELIVERED | Regional | 12 | 1 |
| XXXX050X8731 | JUL-2026 | D-2 | SMS | UNDELIVERED | Regional | 17 | 1 |
| XXXX050X8731 | JUL-2026 | D-2 | VOICE | NO_ANSWER | Hindi | 9 | 1 |
| XXXX050X8731 | JUL-2026 | D-2 | VOICE | NO_ANSWER | Hindi | 14 | 1 |
| XXXX050X8731 | JUL-2026 | D-2 | VOICE_BOT | NO_ANSWER | Hindi | 11 | 1 |
| XXXX050X8731 | JUL-2026 | D-2 | VOICE_BOT | NO_ANSWER | Hindi | 15 | 1 |
| XXXX050X8731 | JUL-2026 | D-3 | SMS | UNDELIVERED | Regional | 17 | 1 |
| XXXX050X8731 | JUL-2026 | D-3 | VOICE | NO_ANSWER | Hindi | 9 | 1 |
| XXXX050X8731 | JUL-2026 | D-3 | VOICE | NO_ANSWER | Hindi | 16 | 1 |
| XXXX050X8731 | JUL-2026 | D-3 | VOICE_BOT | NO_ANSWER | Hindi | 11 | 1 |
| XXXX050X8731 | JUL-2026 | D-3 | VOICE_BOT | NO_ANSWER | Hindi | 15 | 1 |
| XXXX050X8731 | JUL-2026 | D-4 | SMS | UNDELIVERED | English | 12 | 1 |
| XXXX050X8731 | JUL-2026 | D-4 | SMS | UNDELIVERED | Regional | 16 | 1 |
| XXXX050X8731 | JUL-2026 | D-4 | VOICE | NO_ANSWER | English | 17 | 1 |
| XXXX050X8731 | JUL-2026 | D-4 | VOICE | NO_ANSWER | Hindi | 15 | 1 |
| XXXX050X8731 | JUL-2026 | D-4 | VOICE_BOT | NO_ANSWER | Hindi | 11 | 1 |
| XXXX050X8731 | JUL-2026 | D-4 | VOICE_BOT | NO_ANSWER | Hindi | 14 | 1 |
| XXXX050X8731 | JUL-2026 | D-5 | SMS | UNDELIVERED | English | 16 | 1 |
| XXXX050X8731 | JUL-2026 | D-5 | SMS | UNDELIVERED | Regional | 12 | 1 |
| XXXX050X8731 | JUL-2026 | D-5 | VOICE | NO_ANSWER | Hindi | 15 | 1 |
| XXXX050X8731 | JUL-2026 | D-5 | VOICE | NO_ANSWER | Hindi | 17 | 1 |
| XXXX050X8731 | JUL-2026 | D-5 | VOICE_BOT | NO_ANSWER | Hindi | 11 | 1 |
| XXXX050X8731 | JUN-2026 | D-1 | SMS | UNDELIVERED | English | 17 | 1 |
| XXXX050X8731 | JUN-2026 | D-1 | SMS | UNDELIVERED | Regional | 9 | 1 |

## Business Explanation by Observation

### LOW Risk

- `D-5`: The 182 blank APACs are cases where no-campaign scored highest. These APACs still have history, but their feature pattern was closer to the historical no-campaign class. Voice Bot appears rarely because only a small number of LOW-risk APACs had a strong enough Voice Bot signal to beat SMS/IVR/no-campaign.
- `D-4`: If only a small number of APACs received a strategy and the channel is mainly IVR, that means the D-4 model found IVR patterns stronger than SMS for those APACs. For other LOW-risk APACs, no-campaign or non-SMS options ranked higher.
- `D-3`: Blank recommendations mean no-campaign won top-1 for those APACs. IVR and Voice Bot appear because their historical connected/interaction signals were stronger than SMS for those accounts.
- `D-2`: Only a small number of SMS recommendations means SMS did not score highest for most LOW-risk APACs on this day. Either no-campaign or another channel had a higher probability.
- `D-1`: Fewer blanks indicate the model found stronger pre-due final-reminder signals for more APACs compared with earlier days.

### MEDIUM and HIGH Risk

- MEDIUM/HIGH risks can show more than one strategy because the ranking policy allows more recommendations than LOW risk.
- SMS language bifurcation still depends on source-data language quality. If SMS rows mostly carry `Regional`/`English`, output will also mostly show those values.
- Voice Bot counts are low where only a few APACs have enough Voice Bot success/intensity to rank it in the top recommendations.

## Conclusion

The observations are mostly explainable from model ranking and source-data distribution, not from missing prediction generation. The key operational checks are:

- Confirm SMS `verbiage_language` is populated with the expected granular languages, not only `Regional`.
- Confirm whether LOW risk should continue to expose only top-1 or should expose alternates when top-1 is `-`.
- Confirm whether business wants to suppress no-campaign for D-5/D-4, because currently `-` is a valid learned class.

## Detailed Data Evidence for LOW Risk D-5 No-Campaign

### Why Global Top Label Does Not Always Win

`SMS-11AM-ENGLISH` is the most common LOW-risk D-5 training label globally, but the model does not assign the global majority label to every APAC. It compares each APAC feature pattern against learned historical patterns. For the 182 APACs, the D-5 model scored `-` higher than all campaign classes.

### Group-Level Feature Comparison

| Metric | D-5 Blank Group Average | D-5 Campaign Group Average | Business Reading |
|---|---:|---:|---|
| SMS successful signals | 9.90 | 12.35 | Blank group has weaker SMS success. |
| SMS failed signals | 5.36 | 1.41 | Blank group has much higher SMS failure. |
| SMS total attempts | 30.10 | 32.28 |  |
| Voice successful signals | 9.70 | 9.44 |  |
| Voice failed signals | 19.85 | 22.02 |  |
| Voice total attempts | 29.55 | 31.46 |  |
| Voice Bot successful signals | 8.12 | 8.42 |  |
| Voice Bot failed signals | 11.64 | 12.20 |  |
| Voice Bot total attempts | 19.76 | 20.62 |  |
| Total successful signals | 27.72 | 30.21 |  |
| Total failed signals | 36.85 | 35.63 |  |
| Total communication attempts | 79.41 | 84.36 | Both groups have enough history; blanks are not caused by missing data. |
| Overall historical success rate % | 37.24 | 36.58 | Overall rate is similar, so channel/time/language pattern matters more than only total success rate. |

### Important Difference Columns

These are feature columns where the average value differs most between LOW-risk D-5 blank cases and LOW-risk D-5 campaign cases.

| Feature | Blank Avg | Campaign Avg | Difference |
|---|---:|---:|---:|
| `SMS_TOTAL_INTENSITY_M2` | 4.549 | 6.606 | 2.057 |
| `SMS_TOTAL_INTENSITY_M1` | 4.198 | 6.023 | 1.825 |
| `VOICE_TOTAL_INTENSITY_M1` | 5.423 | 7.059 | 1.636 |
| `VOICE_TOTAL_INTENSITY_M2` | 4.720 | 6.118 | 1.398 |
| `VOICE_FAILED_TAMIL_M1` | 0.511 | 1.770 | 1.259 |
| `VOICE_BOT_TOTAL_INTENSITY_M1` | 4.291 | 5.445 | 1.154 |
| `VOICE_BOT_TOTAL_INTENSITY_M2` | 4.451 | 5.425 | 0.974 |
| `SMS_FAILED_REGIONAL_M1` | 1.297 | 0.391 | -0.905 |
| `SMS_TOTAL_INTENSITY_M4` | 7.077 | 6.220 | -0.857 |
| `SMS_TOTAL_INTENSITY_M5` | 4.956 | 4.169 | -0.787 |
| `VOICE_TOTAL_INTENSITY_M5` | 4.791 | 4.056 | -0.735 |
| `SMS_FAILED_REGIONAL_M2` | 0.929 | 0.197 | -0.732 |
| `SMS_FAILED_REGIONAL_M3` | 0.901 | 0.184 | -0.717 |
| `SMS_TOTAL_INTENSITY_M6` | 2.467 | 1.762 | -0.705 |
| `VOICE_TOTAL_INTENSITY_M3` | 5.863 | 6.535 | 0.672 |
| `VOICE_FAILED_TAMIL_M2` | 0.280 | 0.949 | 0.669 |
| `VOICE_TOTAL_INTENSITY_M6` | 2.462 | 1.808 | -0.653 |
| `VOICE_BOT_FAILED_HINDI_M3` | 2.467 | 1.852 | -0.615 |
| `VOICE_FAILED_ENGLISH_M3` | 0.346 | 0.962 | 0.615 |
| `VOICE_FAILED_TAMIL_M3` | 0.269 | 0.821 | 0.552 |

### Sample APAC-Level Proof

| APAC | D-5 Prediction | SMS Success | SMS Failed | SMS Total | Voice Success | Voice Failed | Voice Total | Voice Bot Success | Voice Bot Failed | Voice Bot Total | Total Success | Total Failed | Total Intensity | Success Rate % | Top D-5 Model Probabilities |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| XXXX079X4952 | SMS-12PM-REGIONAL | 18.00 | 0.00 | 45.00 | 11.00 | 38.00 | 49.00 | 8.00 | 16.00 | 24.00 | 37.00 | 54.00 | 118.00 | 31.36 | SMS-12PM-REGIONAL: 0.386247, IVR-4PM-TAMIL: 0.19456, VOICE_BOT-10AM-ENGLISH: 0.12157, -: 0.080078, IVR-3PM-TAMIL: 0.064206 |
| XXXX105X3293 | SMS-3PM-REGIONAL | 16.00 | 10.00 | 50.00 | 2.00 | 47.00 | 49.00 | 5.00 | 29.00 | 34.00 | 23.00 | 86.00 | 133.00 | 17.29 | SMS-3PM-REGIONAL: 0.454289, IVR-4PM-HINDI: 0.144841, -: 0.141225, SMS-12PM-REGIONAL: 0.135299, IVR-9AM-HINDI: 0.056287 |
| XXXX060X1836 | - | 17.20 | 5.00 | 48.00 | 3.00 | 47.00 | 50.00 | 0.00 | 36.00 | 36.00 | 20.20 | 88.00 | 134.00 | 15.07 | -: 0.27114, SMS-12PM-REGIONAL: 0.265136, SMS-3PM-REGIONAL: 0.185396, IVR-9AM-GUJARATI: 0.075843, IVR-4PM-GUJARATI: 0.072181 |
| XXXX060X5639 | - | 12.00 | 0.00 | 30.00 | 10.00 | 21.00 | 31.00 | 7.00 | 13.00 | 20.00 | 29.00 | 34.00 | 81.00 | 35.80 | -: 0.380414, SMS-12PM-REGIONAL: 0.197477, IVR-4PM-MARATHI: 0.119919, SMS-3PM-REGIONAL: 0.08165, IVR-4PM-HINDI: 0.041395 |
| XXXX060X6588 | - | 12.00 | 0.00 | 30.00 | 22.00 | 6.00 | 28.00 | 16.00 | 2.00 | 18.00 | 50.00 | 8.00 | 76.00 | 65.79 | -: 0.148455, SMS-12PM-REGIONAL: 0.14129, IVR-4PM-MARATHI: 0.12444, SMS-3PM-REGIONAL: 0.109539, IVR-3PM-MARATHI: 0.106013 |
| XXXX100X5124 | VOICE_BOT-10AM-ENGLISH | 12.40 | 17.00 | 48.00 | 7.00 | 34.00 | 41.00 | 13.00 | 15.00 | 28.00 | 32.40 | 66.00 | 117.00 | 27.69 | VOICE_BOT-10AM-ENGLISH: 0.213087, IVR-4PM-TAMIL: 0.172807, -: 0.172497, SMS-12PM-REGIONAL: 0.155852, IVR-3PM-TAMIL: 0.103212 |

### Business Conclusion

For the 182 LOW-risk D-5 blank cases, the model did not ignore SMS. It saw SMS history, but the pattern had weaker SMS success and higher SMS failure compared with LOW-risk cases where a campaign was recommended. Since LOW risk only publishes top-1, if `-` wins the probability ranking, the output becomes no campaign. If the business wants to force communication on D-5, that should be added as a business rule, for example: "for D-5, if top-1 is `-`, publish the best non-blank alternate."

