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
        end
        
        subgraph State
            ImgState[Image States]
            MaskState[Mask States]
            PromptState[Prompt Management]
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
            Inference --> PA[Prompt Agent]
        end
    end

    subgraph Core["Core Processing"]
        ImgProc[ImageProcessor]
        QualityMgr[ImageQualityManager]
        ModelSetup[Model Setup]
        
        ImgProc --> QualityMgr
        ImgProc --> |"Style Analysis"| Style[Image Style Analysis]
        ModelSetup --> |"Load Models"| Models[AI Models]
        
        subgraph Processing["Image Processing"]
            Preprocess[Preprocessing]
            Enhance[Enhancement]
            Blend[Image Blending]
            Resize[Resizing]
        end
    end
    
    Pipeline --> ImgProc
    ImgProc --> Processing
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
