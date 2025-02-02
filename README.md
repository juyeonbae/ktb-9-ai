## 아키텍처
```mermaid
graph TB
    subgraph Frontend["Frontend (React)"]
        UI[ImageMaskEditor.jsx]
        UI --> |"HTTP Requests"| API
        UI --> |"Canvas Operations"| Canvas[Canvas Management]
        UI --> |"State Management"| State[Image State]
        
        subgraph Canvas
            Draw[Drawing Tools]
            Mask[Mask Generation]
            History[Edit History]
            BrushTool[Brush Tool]
            RectTool[Rectangle Tool]
            Draw --> BrushTool
            Draw --> RectTool
        end
        
        subgraph State
            ImgState[Image States]
            MaskState[Mask States]
            PromptState[Prompt Management]
            ImageStateManager[ImageStateManager]
            ImageStateManager --> ImgState
            ImageStateManager --> MaskState
        end
    end

    subgraph Backend["Backend (FastAPI)"]
        API["/api/edit-image Endpoint"] 
        API --> Pipeline
        API --> ErrorHandle[Error Handling]
        
        subgraph Pipeline["Image Processing Pipeline"]
            Inference[ImageEditPipeline]
            Inference --> |"Object Removal"| Lama[LaMa Cleaner]
            Inference --> |"Object Generation"| SD[Stable Diffusion]
            Inference --> PA[PromptOptimizationAgent]
            PA --> |"Style Analysis"| StyleAnalysis[Image Style Analysis]
        end
    end

    subgraph Core["Core Processing"]
        ImgProc[ImageProcessor]
        QualityMgr[ImageQualityManager]
        DeviceUtils[Device Utils]
        
        ImgProc --> QualityMgr
        ModelSetup[Model Setup]
        ModelSetup --> |"Load Models"| Models[AI Models]
        ModelSetup --> DeviceUtils
        
        subgraph Processing["Image Processing"]
            Preprocess[Preprocessing]
            Enhance[Enhancement]
            Blend[Image Blending]
            Resize[Resizing]
            MemoryMgr[CUDA Memory Manager]
        end
    end
    
    Pipeline --> ImgProc
    ImgProc --> Processing
```


```mermaid
sequenceDiagram
    actor User
    participant API
    participant Processor
    participant Model
    participant QualityManager
    User->>API: 이미지 업로드
    API->>Processor: 처리 요청
    Processor->>QualityManager: 품질 체크
    QualityManager-->>Processor: 품질 스코어
    
    alt 품질 개선 필요
        Processor->>Model: 이미지 개선
        Model-->>Processor: 개선된 이미지
    end
    
    Processor->>Model: 모델 추론
    Model-->>Processor: 처리 결과
    Processor->>QualityManager: 후처리
    QualityManager-->>Processor: 최종 결과
    Processor-->>API: 응답
    API-->>User: 결과 전달
```

## Build 방법

### 가상환경 생성 및 활성화
```bash
    python3 -m venv venv
    # Windows
    venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
```

### requirements.txt 설치
```bash
    pip install -r requirements.txt
```
<br>

## 로컬 테스트 환경 설정 가이드

### OpenAI API Key 발급

1. 아래 링크에서 OpenAI API Key를 발급받습니다:  
   [OpenAI API Key 발급 페이지](https://platform.openai.com/api-keys)

---

### 환경 변수 파일 생성 및 API Key 설정

1. 터미널에서 프로젝트 루트 디렉토리로 이동 후, 아래 명령어를 실행합니다:
   ```bash
   touch .env
   ```

2. 생성된 .env 파일을 열고 내용 추가:
   ```bash
   OPENAI_API_KEY='발급받은 키를 여기에 복사 붙여넣기'
   ```

----
### 실행
   ```bash
   python main.py
   npm start
   ```
