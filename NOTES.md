Некоторые результаты попыток запуска модели на GTX 1080 (8Gb)

| OCR Model | Paper ArXiv.IDX | Page count | Total time | Average time | Max GPU Mem(Gib) | Comment | Log file |
|-----------|-----------------|------------|------------|--------------|----------------------------|---------|----------|
| Deepseek  | 1606.08693      | 9          | 348.03     |38.49         |9+/8 |Forgot enable quant-config|[log](logs/ocr_15112025_1522.log-saved)|
| Deepseek  | 1606.08693      | 9          | 608.18     |67.22         |6/8   |4bit + max mem limit|[log](logs/ocr_15112025_1643.log-saved)|
| Deepseek  | 1606.08693      | 9          | 600.45     |66.37         |6/8   |4bit, no limit|[log](logs/ocr_15112025_1701.log-saved)|
| Deepseek  | 2511.03951v1    | 24         | 2053       |84            |4.8/8 |4bit, max mem, somehow worked|[log](logs/deepseek_ocr_16112025_2149.log-saved) |
