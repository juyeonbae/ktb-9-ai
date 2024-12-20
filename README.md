## 아키텍처
```mermaid
flowchart TB
    subgraph Frontend
        Client[Web Client]
    end

    subgraph Backend
        API[FastAPI Server]
        Agent[LLM Agent]
        Models[AI Models]
        Storage[Image Storage]
    end

    subgraph ExternalServices
        TritonServer[Triton Inference Server]
        TranslationService[Translation Service]
    end

    Client -->|HTTP Requests| API
    API -->|Process Requests| Agent
    Agent -->|Model Inference| Models
    API -->|Image Processing| TritonServer
    Agent -->|Text Translation| TranslationService
    Models -->|Save Results| Storage
    Storage -->|Return Results| API
    API -->|HTTP Responses| Client
```

```mermaid
classDiagram
    class FastAPIApplication {
        +edit_image()
        +health_check()
    }

    class ImageEditAgent {
        -llm: OpenAI
        -memory: ConversationBufferMemory
        -tools: List[Tool]
        -agent: ZeroShotAgent
        +process_request()
        -_create_tools()
        -_create_agent()
        -_analyze_image()
        -_enhance_prompt()
        -_plan_edits()
    }

    class ImageStateManager {
        -state: Dict
        +backward_inference_image()
        +forward_inference_image()
        +reset_inference_image()
        +reset_text()
        +reset_coord()
        +get_current_image()
        +add_image()
        +get_state()
    }

    class ImageQualityManager {
        -min_resolution: int
        -max_resolution: int
        -quality_threshold: float
        +check_image_quality()
        +enhance_image()
        +resize_if_needed()
        +process_image()
    }

    class PromptProcessor {
        -translation_manager: TranslationManager
        -action_map: Dict
        -object_map: Dict
        -object_details: Dict
        +process()
        +process_generation_prompt()
        -_enhance_prompt()
    }

    class ModelSetup {
        +get_triton_client()
        +get_sd_inpaint()
        +get_lama_cleaner()
        +get_instruct_pix2pix()
    }

    class Utils {
        +save_uploaded_image()
        +save_uploaded_file()
        +save_dataframe()
        +resize_image()
        +plot_bboxes()
        +combine_masks()
    }

    FastAPIApplication --> ImageEditAgent
    FastAPIApplication --> ImageStateManager
    FastAPIApplication --> ImageQualityManager
    ImageEditAgent --> PromptProcessor
    ImageEditAgent --> ImageQualityManager
    FastAPIApplication --> ModelSetup
    FastAPIApplication --> Utils
    PromptProcessor --> TranslationManager
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