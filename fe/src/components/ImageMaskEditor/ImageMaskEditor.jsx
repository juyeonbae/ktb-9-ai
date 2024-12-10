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

  useEffect(() => {
    if (image) {
      const img = new Image();
      img.onload = () => {
        // 원본 크기의 캔버스 설정
        const canvas = canvasRef.current;
        const maskCanvas = maskCanvasRef.current;
        const displayCanvas = displayCanvasRef.current;
        
        // 원본 크기 유지
        canvas.width = img.width;
        canvas.height = img.height;
        maskCanvas.width = img.width;
        maskCanvas.height = img.height;
        
        // 화면에 표시될 크기 계산
        const maxWidth = window.innerWidth * 0.8;
        const maxHeight = window.innerHeight * 0.8;
        const scale = Math.min(maxWidth / img.width, maxHeight / img.height);
        setScale(scale);
        
        // 디스플레이 캔버스 크기 설정
        displayCanvas.width = img.width * scale;
        displayCanvas.height = img.height * scale;
        
        // 이미지 그리기
        const ctx = canvas.getContext('2d');
        const displayCtx = displayCanvas.getContext('2d');
        const maskCtx = maskCanvas.getContext('2d');
        
        ctx.drawImage(img, 0, 0);
        displayCtx.drawImage(img, 0, 0, displayCanvas.width, displayCanvas.height);
        
        maskCtx.fillStyle = 'black';
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
      };
      img.src = URL.createObjectURL(image);
    }
  }, [image]);

  const draw = (e) => {
    if (!isDrawing) return;

    const displayCanvas = displayCanvasRef.current;
    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    const rect = displayCanvas.getBoundingClientRect();
    
    // 마우스 위치를 원본 크기로 변환
    const x = (e.clientX - rect.left) / scale;
    const y = (e.clientY - rect.top) / scale;

    const ctx = canvas.getContext('2d');
    const displayCtx = displayCanvas.getContext('2d');
    const maskCtx = maskCanvas.getContext('2d');

    const actualBrushSize = brushSize / scale;

    // 마스크에 그리기
    maskCtx.fillStyle = 'white';
    maskCtx.beginPath();
    maskCtx.arc(x, y, actualBrushSize / 2, 0, Math.PI * 2);
    maskCtx.fill();

    // 표시용 오버레이 그리기
    displayCtx.fillStyle = 'rgba(255, 0, 0, 0.3)';
    displayCtx.beginPath();
    displayCtx.arc(x * scale, y * scale, brushSize / 2, 0, Math.PI * 2);
    displayCtx.fill();
  };

  const handleSubmit = async () => {
    if (!image || !prompt.trim()) {
      setError('Please select an image and enter a prompt');
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      const maskCanvas = maskCanvasRef.current;
      const maskData = maskCanvas.toDataURL('image/png').split(',')[1];

      const formData = new FormData();
      formData.append('image', image);
      formData.append('prompt', prompt.trim());
      formData.append('mask_data', maskData);

      const response = await fetch('http://localhost:8000/api/edit-image', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(await response.text());
      }

      const blob = await response.blob();
      setImage(blob);
    } catch (error) {
      console.error('Error:', error);
      setError(error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const startDrawing = (e) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
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

      <button
        onClick={handleSubmit}
        disabled={!image || isLoading}
        className={`px-4 py-2 rounded ${
          image && !isLoading
            ? 'bg-blue-500 text-white hover:bg-blue-600' 
            : 'bg-gray-300 text-gray-500'
        }`}
      >
        {isLoading ? 'Processing...' : 'Edit Image'}
      </button>
    </div>
  );
};

export default ImageMaskEditor;