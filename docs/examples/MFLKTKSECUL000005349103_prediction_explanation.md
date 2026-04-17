# Strategy prediction explanation: MFLKTKSECUL000005349103

## Account Summary

| APAC_CARD_NUMBER | SOURCE_RISK | SOURCE_MONTH_USED | PREDICT_MONTH | feature_months | schedule_months | sms_total | wh_total | voice_total | success_total | failed_total | nonblank_schedule_slots | all_blank_schedule_months |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MFLKTKSECUL000005349103 | LOW | APR-2026 | MAY-2026 | FEB-2026,MAR-2026,APR-2026 | FEB-2026,MAR-2026,APR-2026 | 11 | 11 | 11 | 20 | 13 | 13 | APR-2026 |

## Recent Schedule History

FEB-2026=SMS-9AM-REGIONAL;WH-9AM-KANNADA;SMS-4PM-ENGLISH;SMS-8AM-REGIONAL;SMS-5PM-ENGLISH;SMS-10AM-REGIONAL... | MAR-2026=SMS-11AM-REGIONAL;WH-3PM-ENGLISH;SMS-4PM-ENGLISH;WH-5PM-ENGLISH;WH-1PM-ENGLISH;SMS-3PM-REGIONAL | APR-2026=ALL_BLANK

## Day-Level Explanation

| day | prediction | top_probability | blank_probability | best_nonblank_label | best_nonblank_probability | top_candidates | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D-5 | - | 0.6297 | 0.6297 | SMS-9AM-REGIONAL | 0.3655 | -=0.6297; SMS-9AM-REGIONAL=0.3655; SMS-12PM-ENGLISH=0.0009; SMS-6PM-ENGLISH=0.0004; SMS-10AM-ENGLISH=0.0004 | Predicted no-contact because blank/no-contact has the highest model probability; LOW shows only top 1. |
| D-4 | - | 0.7266 | 0.7266 | SMS-9AM-ENGLISH | 0.2354 | -=0.7266; SMS-9AM-ENGLISH=0.2354; IVR-2PM-ENGLISH=0.0357; WH-8AM-KANNADA=0.0002; WH-9AM-KANNADA=0.0002 | Predicted no-contact because blank/no-contact has the highest model probability; LOW shows only top 1. |
| D-3 | - | 0.5326 | 0.5326 | WH-9AM-KANNADA | 0.4574 | -=0.5326; WH-9AM-KANNADA=0.4574; WH-9AM-HINDI=0.0024; WH-9AM-TAMIL=0.0013; WH-9AM-TELUGU=0.0013 | Predicted no-contact because blank/no-contact has the highest model probability; LOW shows only top 1. |
| D-2 | - | 0.9697 | 0.9697 | IVR-2PM-ENGLISH | 0.0156 | -=0.9697; IVR-2PM-ENGLISH=0.0156; SMS-9AM-ENGLISH=0.0094; WH-4PM-ENGLISH=0.0014; SMS-3PM-ENGLISH=0.0006 | Predicted no-contact because blank/no-contact has the highest model probability; LOW shows only top 1. |
| D-1 | - | 1.0 | 1.0 | SMS-12PM-ENGLISH | 0.0 | -=1.0000; SMS-12PM-ENGLISH=0.0000; SMS-11AM-ENGLISH=0.0000; SMS-10AM-ENGLISH=0.0000; SMS-3PM-ENGLISH=0.0000 | Predicted no-contact because blank/no-contact has the highest model probability; LOW shows only top 1. |
| D+1 | - | 0.5278 | 0.5278 | SMS-4PM-ENGLISH | 0.4395 | -=0.5278; SMS-4PM-ENGLISH=0.4395; WH-9AM-ENGLISH=0.0245; IVR-11AM-ENGLISH=0.0025; SMS-11AM-ENGLISH=0.0011 | Predicted no-contact because blank/no-contact has the highest model probability; LOW shows only top 1. |
| D+2 | - | 0.5168 | 0.5168 | SMS-8AM-REGIONAL | 0.4143 | -=0.5168; SMS-8AM-REGIONAL=0.4143; WH-5PM-KANNADA=0.0497; WH-5PM-HINDI=0.0034; SMS-5PM-ENGLISH=0.0031 | Predicted no-contact because blank/no-contact has the highest model probability; LOW shows only top 1. |
| D+3 | - | 0.638 | 0.638 | SMS-5PM-ENGLISH | 0.3324 | -=0.6380; SMS-5PM-ENGLISH=0.3324; WH-5PM-ENGLISH=0.0258; WH-9AM-ENGLISH=0.0009; IVR-5PM-ENGLISH=0.0006 | Predicted no-contact because blank/no-contact has the highest model probability; LOW shows only top 1. |
| D+4 | - | 0.5034 | 0.5034 | SMS-10AM-REGIONAL | 0.4595 | -=0.5034; SMS-10AM-REGIONAL=0.4595; WH-3PM-KANNADA=0.0306; WH-3PM-HINDI=0.0013; WH-3PM-TELUGU=0.0011 | Predicted no-contact because blank/no-contact has the highest model probability; LOW shows only top 1. |
| D+5 | - | 0.5931 | 0.5931 | SMS-3PM-ENGLISH | 0.3541 | -=0.5931; SMS-3PM-ENGLISH=0.3541; WH-9AM-ENGLISH=0.0482; IVR-12PM-ENGLISH=0.0014; SMS-11AM-ENGLISH=0.0006 | Predicted no-contact because blank/no-contact has the highest model probability; LOW shows only top 1. |
