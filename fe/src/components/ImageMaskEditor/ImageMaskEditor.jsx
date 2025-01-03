import React, { useEffect, useRef, useState } from "react";

const ImageMaskEditor = () => {
  // State 관리
  const [image, setImage] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [isDrawing, setIsDrawing] = useState(false);
  const [selectedTool, setSelectedTool] = useState("brush");
  const [brushSize, setBrushSize] = useState(20);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [imageHistory, setImageHistory] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [scale, setScale] = useState(1);

  // Canvas Refs
  const canvasRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const displayCanvasRef = useRef(null);

  // State 관리 부분에 아래 두 state 추가
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [currentPos, setCurrentPos] = useState({ x: 0, y: 0 });

  // 이미지 로드 및 초기 설정
  useEffect(() => {
    if (image) {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        const maskCanvas = maskCanvasRef.current;
        if (!maskCanvas) {
          console.error("Mask canvas is not initialized.");
          return;
        }
        const displayCanvas = displayCanvasRef.current;

        // 캔버스 크기 설정
        canvas.width = img.width;
        canvas.height = img.height;
        maskCanvas.width = img.width;
        maskCanvas.height = img.height;

        // 화면에 맞게 스케일 조정
        const maxWidth = window.innerWidth * 0.8;
        const maxHeight = window.innerHeight * 0.8;
        const scale = Math.min(maxWidth / img.width, maxHeight / img.height);
        setScale(scale);

        displayCanvas.width = img.width * scale;
        displayCanvas.height = img.height * scale;

        // 초기 이미지 그리기
        const ctx = canvas.getContext("2d");
        const displayCtx = displayCanvas.getContext("2d");
        const maskCtx = maskCanvas.getContext("2d");

        ctx.drawImage(img, 0, 0);
        displayCtx.drawImage(
          img,
          0,
          0,
          displayCanvas.width,
          displayCanvas.height
        );

        // 마스크 초기화
        maskCtx.fillStyle = "black";
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

        // 히스토리 초기화
        if (!imageHistory.length) {
          setImageHistory([img.src]);
          setCurrentIndex(0);
        }
      };
      img.src = URL.createObjectURL(image);
    }
  }, [image, imageHistory]);

  // 마스크 그리기 함수들
  const getScaledCoordinates = (e) => {
    const displayCanvas = displayCanvasRef.current;
    const rect = displayCanvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) / scale,
      y: (e.clientY - rect.top) / scale,
    };
  };

  const drawBrush = (x, y) => {
    const maskCanvas = maskCanvasRef.current;
    const displayCanvas = displayCanvasRef.current;
    const maskCtx = maskCanvas.getContext("2d");
    const displayCtx = displayCanvas.getContext("2d");
    const actualBrushSize = brushSize / scale;

    maskCtx.fillStyle = "white";
    maskCtx.beginPath();
    maskCtx.arc(x, y, actualBrushSize / 2, 0, Math.PI * 2);
    maskCtx.fill();

    displayCtx.fillStyle = "rgba(255, 0, 0, 0.3)";
    displayCtx.beginPath();
    displayCtx.arc(x * scale, y * scale, brushSize / 2, 0, Math.PI * 2);
    displayCtx.fill();
  };

  const drawRectangle = (startPos, currentPos) => {
    const displayCanvas = displayCanvasRef.current;
    const displayCtx = displayCanvas.getContext("2d");

    // 원본 이미지 복원
    displayCtx.drawImage(
      canvasRef.current,
      0,
      0,
      displayCanvas.width,
      displayCanvas.height
    );

    // 사각형 그리기
    displayCtx.fillStyle = "rgba(255, 0, 0, 0.3)";
    const x = Math.min(startPos.x, currentPos.x);
    const y = Math.min(startPos.y, currentPos.y);
    const width = Math.abs(currentPos.x - startPos.x);
    const height = Math.abs(currentPos.y - startPos.y);

    displayCtx.fillRect(x * scale, y * scale, width * scale, height * scale);
  };

  // 마스크 그리기 이벤트 핸들러
  const startDrawing = (e) => {
    setIsDrawing(true);
    const pos = getScaledCoordinates(e);
    if (selectedTool === "brush") {
      drawBrush(pos.x, pos.y);
    } else {
      setStartPos(pos);
    }
  };

  const draw = (e) => {
    if (!isDrawing) return;

    const pos = getScaledCoordinates(e);
    setCurrentPos(pos); // currentPos 업데이트

    if (selectedTool === "brush") {
      drawBrush(pos.x, pos.y);
    } else {
      drawRectangle(startPos, pos);
    }
  };
  const stopDrawing = () => {
    if (isDrawing && selectedTool === "rectangle") {
      // 마스크에 최종 사각형 그리기
      const maskCanvas = maskCanvasRef.current;
      const maskCtx = maskCanvas.getContext("2d");
      maskCtx.fillStyle = "white";
      const width = Math.abs(currentPos.x - startPos.x);
      const height = Math.abs(currentPos.y - startPos.y);
      maskCtx.fillRect(
        Math.min(startPos.x, currentPos.x),
        Math.min(startPos.y, currentPos.y),
        width,
        height
      );
    }
    setIsDrawing(false);
  };

  const handleEditImage = async (action) => {
    if (!image) {
      setError("Please select an image");
      return;
    }

    if (action === "edit" && (!prompt || prompt.trim().length === 0)) {
      setError("Prompt cannot be empty for editing");
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      // 먼저 마스크 blob 생성
      const maskBlob = await new Promise((resolve, reject) =>
        maskCanvasRef.current.toBlob((blob) => {
          if (blob) resolve(blob);
          else reject(new Error("Failed to create mask blob"));
        }, "image/png")
      );

      if (!maskBlob || maskBlob.size === 0) {
        throw new Error("Mask blob is empty or not created properly.");
      }

      const formData = new FormData();
      formData.append("image", image);
      formData.append("type", action);
      formData.append("mask", maskBlob);

      if (action === "edit") {
        formData.append("prompt", prompt.trim());
      }

      // 디버깅을 위한 로그
      console.log("Image type:", typeof image, image instanceof File);
      console.log("Image object:", image);
      console.log("Mask blob type:", typeof maskBlob);
      console.log("Form Data:");
      formData.forEach((value, key) => console.log(`${key}:`, value));

      const response = await fetch("http://localhost:5002/api/edit-image", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Failed to edit image: ${errorData}`);
      }

      const result = await response.blob();
      const imageUrl = URL.createObjectURL(result);
      setImageHistory((prev) => [...prev.slice(0, currentIndex + 1), imageUrl]);
      setCurrentIndex((prev) => prev + 1);
      setImage(result);
    } catch (error) {
      console.error("Error in handleEditImage:", error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-4 p-4 items-center">
      {/* 상단 컨트롤 */}
      <div className="flex gap-4 items-center mb-4">
        {/* 이미지 업로드 */}
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (e.target.files?.[0]) {
              setImage(e.target.files[0]);
              setPrompt("");
              setImageHistory([]);
              setCurrentIndex(0);
              setError(null);
            }
          }}
          className="p-2 border rounded"
        />

        {/* 프롬프트 입력 */}
        <input
          type="text"
          placeholder="Enter prompt for image editing"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="p-2 border rounded w-64"
        />
      </div>

      {/* 도구 선택 */}
      <div className="flex gap-4 items-center mb-4">
        <div className="flex gap-2">
          <button
            onClick={() => setSelectedTool("brush")}
            className={`px-3 py-1 rounded ${
              selectedTool === "brush"
                ? "bg-blue-500 text-white"
                : "bg-gray-200 hover:bg-gray-300"
            }`}
          >
            Brush
          </button>
          <button
            onClick={() => setSelectedTool("rectangle")}
            className={`px-3 py-1 rounded ${
              selectedTool === "rectangle"
                ? "bg-blue-500 text-white"
                : "bg-gray-200 hover:bg-gray-300"
            }`}
          >
            Rectangle
          </button>
        </div>

        {selectedTool === "brush" && (
          <div className="flex items-center gap-2">
            <span>Brush Size:</span>
            <input
              type="range"
              min="1"
              max="50"
              value={brushSize}
              onChange={(e) => setBrushSize(Number(e.target.value))}
              className="w-32"
            />
          </div>
        )}
      </div>

      {/* 에러 메시지 */}
      {error && <div className="text-red-500 mb-4">{error}</div>}

      {/* 캔버스 */}
      <div className="relative border border-gray-300 rounded">
        <canvas
          ref={displayCanvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseOut={stopDrawing}
          className="cursor-crosshair"
        />
        <canvas ref={canvasRef} className="hidden" />
        <canvas ref={maskCanvasRef} className="hidden" />
      </div>

      {/* 하단 컨트롤 */}
      <div className="flex gap-4">
        {/* Remove/Edit 버튼 */}
        <button
          onClick={() => handleEditImage("remove")}
          disabled={!image || isLoading}
          className={`px-4 py-2 rounded ${
            image && !isLoading
              ? "bg-red-500 text-white hover:bg-red-600"
              : "bg-gray-300 text-gray-500"
          }`}
        >
          Remove Object
        </button>
        <button
          onClick={() => handleEditImage("edit")}
          disabled={!image || isLoading || !prompt.trim()}
          className={`px-4 py-2 rounded ${
            image && !isLoading && prompt.trim()
              ? "bg-blue-500 text-white hover:bg-blue-600"
              : "bg-gray-300 text-gray-500"
          }`}
        >
          Edit Image
        </button>

        {/* History 컨트롤 */}
        <button
          onClick={() => {
            if (currentIndex > 0) {
              setCurrentIndex((prev) => prev - 1);
              const img = new Image();
              img.onload = async () => {
                const blob = await fetch(img.src).then((r) => r.blob());
                setImage(blob);
              };
              img.src = imageHistory[currentIndex - 1];
            }
          }}
          disabled={currentIndex === 0}
          className={`px-4 py-2 rounded ${
            currentIndex > 0
              ? "bg-gray-500 text-white hover:bg-gray-600"
              : "bg-gray-300 text-gray-500"
          }`}
        >
          Previous
        </button>
        <button
          onClick={() => {
            if (currentIndex < imageHistory.length - 1) {
              setCurrentIndex((prev) => prev + 1);
              const img = new Image();
              img.onload = async () => {
                const blob = await fetch(img.src).then((r) => r.blob());
                setImage(blob);
              };
              img.src = imageHistory[currentIndex + 1];
            }
          }}
          disabled={currentIndex >= imageHistory.length - 1}
          className={`px-4 py-2 rounded ${
            currentIndex < imageHistory.length - 1
              ? "bg-gray-500 text-white hover:bg-gray-600"
              : "bg-gray-300 text-gray-500"
          }`}
        >
          Next
        </button>
        <button
          onClick={() => {
            if (imageHistory.length > 0) {
              setCurrentIndex(0);
              setImageHistory([imageHistory[0]]);
              const img = new Image();
              img.onload = async () => {
                const blob = await fetch(img.src).then((r) => r.blob());
                setImage(blob);
              };
              img.src = imageHistory[0];
            }
          }}
          disabled={imageHistory.length <= 1}
          className={`px-4 py-2 rounded ${
            imageHistory.length > 1
              ? "bg-yellow-500 text-white hover:bg-yellow-600"
              : "bg-gray-300 text-gray-500"
          }`}
        >
          Reset
        </button>
      </div>

      {/* Loading 상태 */}
      {isLoading && (
        <div className="fixed top-0 left-0 w-full h-full flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white p-4 rounded">Processing...</div>
        </div>
      )}
    </div>
  );
};

export default ImageMaskEditor;
