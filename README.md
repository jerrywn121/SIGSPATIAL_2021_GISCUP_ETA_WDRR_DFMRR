# SIGSPATIAL_2021_GISCUP_ETA_WDR_DEEPFM
Codes for the [SIGSPATIAL 2021 GISCUP ETA Contest] (https://www.biendata.xyz/competition/didi-eta/)

## Models
- DeepFM
- WDR

## Experiments
- baseline Wide-Deep-Recurrent (WDR) from Didi's paper [Learning Learning to Estimate the Travel Time] (https://dl.acm.org/doi/abs/10.1145/3219819.3219900)

- WDR
  1. Bidirectional LSTM (Bi-LSTM)
  2. Initialize hidden states of Bi-LSTM using embeddings of non-link-cross features, which is also the input to "deep"
  3. Axuiliary LSTM loss (predict link status at arrival time)

- DeepFM
  1. Bidirectional LSTM (Bi-LSTM)
  2. Initialize hidden states of Bi-LSTM using embeddings of non-link-cross features, which is also the input to "deep"
  3. Axuiliary LSTM loss (predict link status at arrival time)

## Result
|    | WDR | DeepFM |
|---|---|---|
| no Bi-LSTM | 0.1295  | 0.1323 |
| Bi-LSTM (1) | 0.1318  | 0.1325 |
| Bi-LSTM + initialize LSTM (2)  | 0.1290  | 0.1292 |
| Bi-LSTM + initialize LSTM (2) + auxiliary loss (3)  | 0.1279  | 0.1277 |




