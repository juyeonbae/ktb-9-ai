import React, { useEffect, useRef, useState } from 'react';

const ImageMaskEditor = () => {
  const canvasRef = useRef(null);
  const maskCanvasRef = useRef(null);
  const displayCanvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(20);
  const [image, setImage] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [scale, setScale] = useState(1);
  const [imageHistory, setImageHistory] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  // 도형 도구 관련 상태 추가
  const [selectedTool, setSelectedTool] = useState('brush'); // 'brush', 'rectangle', 'circle'
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [tempCanvas, setTempCanvas] = useState(null);

  useEffect(() => {
    if (image) {
      const img = new Image();
      img.onload = () => {
        const canvas = canvasRef.current;
        const maskCanvas = maskCanvasRef.current;
        const displayCanvas = displayCanvasRef.current;
        
        canvas.width = img.width;
        canvas.height = img.height;
        maskCanvas.width = img.width;
        maskCanvas.height = img.height;
        
        const maxWidth = window.innerWidth * 0.8;
        const maxHeight = window.innerHeight * 0.8;
        const scale = Math.min(maxWidth / img.width, maxHeight / img.height);
        setScale(scale);
        
        displayCanvas.width = img.width * scale;
        displayCanvas.height = img.height * scale;
        
        const ctx = canvas.getContext('2d');
        const displayCtx = displayCanvas.getContext('2d');
        const maskCtx = maskCanvas.getContext('2d');
        
        ctx.drawImage(img, 0, 0);
        displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
        
        maskCtx.fillStyle = 'black';
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);

        if (!imageHistory || imageHistory.length === 0) {
          setImageHistory([img.src]);
          setCurrentIndex(0);
        }
      };
      img.src = URL.createObjectURL(image);
    }
  }, [image, imageHistory]);

  const getScaledCoordinates = (e) => {
    const displayCanvas = displayCanvasRef.current;
    const rect = displayCanvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) / scale,
      y: (e.clientY - rect.top) / scale
    };
  };

  const drawBrush = (x, y) => {
    const maskCanvas = maskCanvasRef.current;
    const displayCanvas = displayCanvasRef.current;
    const maskCtx = maskCanvas.getContext('2d');
    const displayCtx = displayCanvas.getContext('2d');
    const actualBrushSize = brushSize / scale;

    maskCtx.fillStyle = 'white';
    maskCtx.beginPath();
    maskCtx.arc(x, y, actualBrushSize / 2, 0, Math.PI * 2);
    maskCtx.fill();

    displayCtx.fillStyle = 'rgba(255, 0, 0, 0.3)';
    displayCtx.beginPath();
    displayCtx.arc(x * scale, y * scale, brushSize / 2, 0, Math.PI * 2);
    displayCtx.fill();
  };

  const drawShape = (currentPos) => {
    const displayCanvas = displayCanvasRef.current;
    const displayCtx = displayCanvas.getContext('2d');
  
    // 원본 이미지 다시 그리기
    displayCtx.drawImage(canvasRef.current, 0, 0, displayCanvas.width, displayCanvas.height);
  
    // 좌표 보정
    let x = Math.min(startPos.x, currentPos.x);
    let y = Math.min(startPos.y, currentPos.y);
    let width = Math.abs(currentPos.x - startPos.x);
    let height = Math.abs(currentPos.y - startPos.y);
  
    displayCtx.fillStyle = 'rgba(255, 0, 0, 0.3)';
    if (selectedTool === 'rectangle') {
      displayCtx.fillRect(
        x * scale,
        y * scale,
        width * scale,
        height * scale
      );
    } else if (selectedTool === 'circle') {
      const radius = Math.sqrt(width * width + height * height);
      displayCtx.beginPath();
      displayCtx.arc(startPos.x * scale, startPos.y * scale, radius * scale, 0, Math.PI * 2);
      displayCtx.fill();
    }
  };

  const finalizeShape = (currentPos) => {
    const maskCanvas = maskCanvasRef.current;
    const maskCtx = maskCanvas.getContext('2d');
  
    // 전체를 black으로 초기화 (보존할 영역)
    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    
    // 좌표 보정
    let x = Math.min(startPos.x, currentPos.x);
    let y = Math.min(startPos.y, currentPos.y);
    let width = Math.abs(currentPos.x - startPos.x);
    let height = Math.abs(currentPos.y - startPos.y);
  
    // 선택 영역을 white로 설정 (변경할 영역)
    maskCtx.fillStyle = 'white';
    if (selectedTool === 'rectangle') {
      maskCtx.fillRect(x, y, width, height);
    } else if (selectedTool === 'circle') {
      const radius = Math.sqrt(width * width + height * height);
      maskCtx.beginPath();
      maskCtx.arc(startPos.x, startPos.y, radius, 0, Math.PI * 2);
      maskCtx.fill();
    }
  };

  const startDrawing = (e) => {
    setIsDrawing(true);
    const pos = getScaledCoordinates(e);
    setStartPos(pos);
    
    if (selectedTool === 'brush') {
      drawBrush(pos.x, pos.y);
    }
  };

  const draw = (e) => {
    if (!isDrawing) return;

    const currentPos = getScaledCoordinates(e);
    
    if (selectedTool === 'brush') {
      drawBrush(currentPos.x, currentPos.y);
    } else {
      drawShape(currentPos);
    }
  };

  const stopDrawing = (e) => {
    if (isDrawing && selectedTool !== 'brush') {
      const currentPos = getScaledCoordinates(e);
      finalizeShape(currentPos);
    }
    setIsDrawing(false);
  };

  const handleSubmit = async () => {
    if (!image || !prompt.trim()) {
      setError('Please select an image and enter a prompt');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const formData = new FormData();
      formData.append('image', image); 

      // 마스크 데이터 품질 유지
      const maskCanvas = maskCanvasRef.current;
      const maskBlob = await new Promise(resolve => {
        maskCanvas.toBlob(resolve, 'image/png', 1.0); // PNG 형식으로 저장하여 품질 손실 방지
      });
      // base64로 인코딩
      const reader = new FileReader();
      const maskData = await new Promise((resolve) => {
        reader.onloadend = () => {
          const base64data = reader.result;
          resolve(base64data.split(',')[1]); // base64 데이터 부분만 추출
        };
        reader.readAsDataURL(maskBlob);
      });
      
      formData.append('mask_data', maskData);
      formData.append('prompt', prompt.trim());

      const response = await fetch('http://localhost:8000/api/edit-image', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const blob = await response.blob();
      const editedImageUrl = URL.createObjectURL(blob);
      
      // 히스토리에 새로운 이미지 추가 (현재 인덱스 이후의 기록은 삭제)
      setImageHistory(prev => [...prev.slice(0, currentIndex + 1), editedImageUrl]);
      setCurrentIndex(prev => prev + 1);
      setImage(blob);
    } catch (error) {
      console.error('Error:', error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePrevious = () => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
      const img = new Image();
      img.onload = async () => {  // async 추가
        const displayCanvas = displayCanvasRef.current;
        const displayCtx = displayCanvas.getContext('2d');
        displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
  
        const blob = await fetch(img.src).then(r => r.blob());
        setImage(blob);
      };
      img.src = imageHistory[currentIndex - 1];
    }
  };
  
  const handleNext = () => {
    if (currentIndex < imageHistory.length - 1) {
      setCurrentIndex(prev => prev + 1);
      const img = new Image();
      img.onload = async () => {  // async 추가
        const displayCanvas = displayCanvasRef.current;
        const displayCtx = displayCanvas.getContext('2d');
        displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
  
        const blob = await fetch(img.src).then(r => r.blob());
        setImage(blob);
      };
      img.src = imageHistory[currentIndex + 1];
    }
  };

  const handleReset = () => {
    if (imageHistory.length > 0) {
      setCurrentIndex(0);
      setImageHistory([imageHistory[0]]); // 첫 번째 이미지만 남기고 모두 제거
  
      // 원본 이미지 상태로 초기화
      const img = new Image();
      img.onload = async () => {  // async 추가
        // 디스플레이 캔버스 초기화
        const displayCanvas = displayCanvasRef.current;
        const displayCtx = displayCanvas.getContext('2d');
        displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
          
        // 메인 이미지 상태 업데이트
        const blob = await fetch(img.src).then(r => r.blob());
        setImage(blob);
          
        // 마스크 캔버스도 초기화
        const maskCanvas = maskCanvasRef.current;
        const maskCtx = maskCanvas.getContext('2d');
        maskCtx.fillStyle = 'black';
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
      };
      img.src = imageHistory[0];
    }
  };


  return (
    <div className="flex flex-col gap-4 p-4 items-center">
      <div className="flex gap-4 items-center mb-4">
        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            if (e.target.files?.[0]) {
              setImage(e.target.files[0]);
              setError(null);
              setPrompt('');
              setImageHistory([]);
              setCurrentIndex(0);
            }
          }}
          className="p-2 border rounded"
        />
        <input
          type="text"
          placeholder="Enter prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="p-2 border rounded w-64"
        />
        
        {/* 도구 선택 버튼 추가 */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setSelectedTool('brush')}
            className={`px-3 py-1 rounded ${
              selectedTool === 'brush'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            Brush
          </button>
          <button
            onClick={() => setSelectedTool('rectangle')}
            className={`px-3 py-1 rounded ${
              selectedTool === 'rectangle'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            Rectangle
          </button>
          <button
            onClick={() => setSelectedTool('circle')}
            className={`px-3 py-1 rounded ${
              selectedTool === 'circle'
                ? 'bg-blue-500 text-white'
                : 'bg-gray-200 hover:bg-gray-300'
            }`}
          >
            Circle
          </button>
        </div>

        {selectedTool === 'brush' && (
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

      {error && <div className="text-red-500 mb-4">{error}</div>}

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

      <div className="flex gap-4">
        <button
          onClick={handlePrevious}
          disabled={currentIndex === 0}
          className={`px-4 py-2 rounded ${
            currentIndex > 0
              ? 'bg-blue-500 text-white hover:bg-blue-600'
              : 'bg-gray-300 text-gray-500'
          }`}
        >
          Previous
        </button>

        <button
          onClick={handleNext}
          disabled={currentIndex >= imageHistory.length - 1}
          className={`px-4 py-2 rounded ${
            currentIndex < imageHistory.length - 1
              ? 'bg-blue-500 text-white hover:bg-blue-600'
              : 'bg-gray-300 text-gray-500'
          }`}
        >
          Next
        </button>

        <button
          onClick={handleReset}
          disabled={imageHistory.length <= 1}
          className={`px-4 py-2 rounded ${
            imageHistory.length > 1
              ? 'bg-yellow-500 text-white hover:bg-yellow-600'
              : 'bg-gray-300 text-gray-500'
          }`}
        >
          Reset
        </button>

        <button
          onClick={handleSubmit}
          disabled={!image || isLoading}
          className={`px-4 py-2 rounded ${
            image && !isLoading
              ? 'bg-green-500 text-white hover:bg-green-600'
              : 'bg-gray-300 text-gray-500'
          }`}
        >
          {isLoading ? 'Processing...' : 'Edit Image'}
        </button>
      </div>
    </div>
  );
};

export default ImageMaskEditor;