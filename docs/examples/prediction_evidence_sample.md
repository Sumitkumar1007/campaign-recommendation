# Prediction Evidence Sample

This sample shows how a recommendation can be justified using the historical feature values that were fed into the model.

## Business View

- Loan Number: `MOB-TEST-AI`
- Prediction Month: `JUL-2026`
- Source Month Used: `JUN-2026`
- Risk: `MEDIUM`
- Vertical: `LAP`
- Recommended Action for Follow-up Day: `SMS at 4PM in English`

## Business Explanation

The recommendation was generated because historical customer communication data shows that SMS has been the strongest channel for this account in the recent history window. Within SMS, the 4PM English slot has shown the highest positive interaction pattern compared with other available options. Based on this pattern, the model selected `SMS-4PM-ENGLISH` as the most suitable recommendation for that day.

## Sample Evidence Payload

```json
{
  "loanNumber": "MOB-TEST-AI",
  "sourceMonthUsed": "JUN-2026",
  "predictionMonth": "JUL-2026",
  "risk": "MEDIUM",
  "vertical": "LAP",
  "recommendedStrategy": "SMS-4PM-ENGLISH",
  "recommendedActionText": "SMS at 4PM in English",
  "channelTotals": {
    "SMS": 12,
    "WH": 4,
    "VOICE": 1,
    "VOICE_BOT": 0
  },
  "supportingFeatures": {
    "SMS_SUCCESS_4PM_ENGLISH": 5,
    "SMS_SUCCESS_9AM_ENGLISH": 2,
    "SMS_FAILED_ENGLISH": 1,
    "WH_SUCCESS_4PM_ENGLISH": 1,
    "WH_SUCCESS_9AM_ENGLISH": 1,
    "VOICE_SUCCESS_4PM_ENGLISH": 0
  },
  "topCompetingStrategies": [
    {
      "strategy": "SMS-4PM-ENGLISH",
      "reason": "Highest historical success pattern within the strongest channel."
    },
    {
      "strategy": "WH-9AM-ENGLISH",
      "reason": "Secondary digital option with lower historical success counts."
    },
    {
      "strategy": "VOICE-2PM-ENGLISH",
      "reason": "Available but weak historical interaction pattern."
    }
  ],
  "businessNarrative": "The model selected SMS at 4PM in English because SMS has the highest communication intensity for this account in the recent history window, and the 4PM English SMS pattern has the strongest historical success signal among the available options."
}
```

## How To Read This

- `channelTotals`: Overall historical interaction intensity by channel.
- `supportingFeatures`: The key feature values that support the chosen recommendation.
- `topCompetingStrategies`: Other likely alternatives that were weaker than the final recommendation.
- `businessNarrative`: A plain-language version suitable for campaign and management teams.

## Suggested Usage

This sample can be shown to business teams as the proposed explanation format for future prediction auditability.
