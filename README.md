# CvHub OCR

## Module Pipeline Process
```
receive request
  |
  | server.py
  ↓ 
load config
  |
  | config/
  ↓ 
image input and preprocess
  |
  | preprocess/
  ↓
build ocr engine
  |
  | detector/ + recognizer/ => engine/
  ↓
engine run
  |
  | engine/
  ↓
postprocess/
  |
  | postprocess/
  ↓
send response -- server.py
```