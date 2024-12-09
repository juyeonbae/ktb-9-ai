import React, { useEffect, useRef, useState } from 'react';

const ImageMaskEditor = () => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [brushSize, setBrushSize] = useState(20);
  const [image, setImage] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [error, setError] = useState(null);
  
  // 별도의 마스크 캔버스 추가
  const maskCanvasRef = useRef(null);

  useEffect(() => {
    if (image) {
      const canvas = canvasRef.current;
      const maskCanvas = maskCanvasRef.current;
      const ctx = canvas.getContext('2d');
      const maskCtx = maskCanvas.getContext('2d');
      
      const img = new Image();
      img.onload = () => {
        // 메인 캔버스 설정
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        // 마스크 캔버스 설정
        maskCanvas.width = img.width;
        maskCanvas.height = img.height;
        maskCtx.fillStyle = 'black';
        maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
      };
      img.src = URL.createObjectURL(image);
    }
  }, [image]);

  const handleImageUpload = (e) => {
    if (e.target.files && e.target.files[0]) {
      setImage(e.target.files[0]);
      setError(null);
    }
  };

  const startDrawing = (e) => {
    setIsDrawing(true);
    draw(e);
  };

  const stopDrawing = () => {
    setIsDrawing(false);
  };

  const draw = (e) => {
    if (!isDrawing) return;

    const canvas = canvasRef.current;
    const maskCanvas = maskCanvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const ctx = canvas.getContext('2d');
    const maskCtx = maskCanvas.getContext('2d');
    
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // 메인 캔버스에 시각적 피드백
    ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
    ctx.beginPath();
    ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    ctx.fill();

    // 마스크 캔버스에 실제 마스크 그리기
    maskCtx.fillStyle = 'white';
    maskCtx.beginPath();
    maskCtx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
    maskCtx.fill();
  };

  const handleSubmit = async () => {
    try {
      setError(null);
      
      // 마스크 데이터 최적화: 압축된 PNG로 변환
      const maskCanvas = maskCanvasRef.current;
      const maskData = maskCanvas.toDataURL('image/png', 0.5).split(',')[1];

      const formData = new FormData();
      formData.append('image', image);
      formData.append('prompt', prompt);
      formData.append('mask_data', maskData);

      console.log('Sending mask data...');

      const response = await fetch('http://localhost:8000/api/edit-image', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setImage(await fetch(imageUrl).then(r => r.blob()));
      } else {
        const errorData = await response.text();
        console.error('Server error:', errorData);
        setError(`Server error: ${errorData}`);
      }
    } catch (error) {
      console.error('Error:', error);
      setError('Failed to process image');
    }
  };

  return (
    <div className="flex flex-col gap-4 p-4 items-center">
      <div className="flex gap-4 items-center mb-4">
        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          className="p-2 border rounded"
        />
        <input
          type="text"
          placeholder="Enter prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="p-2 border rounded"
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

      {error && (
        <div className="text-red-500 mb-4">{error}</div>
      )}

      <div className="relative border border-gray-300 rounded">
        <canvas
          ref={canvasRef}
          onMouseDown={startDrawing}
          onMouseMove={draw}
          onMouseUp={stopDrawing}
          onMouseOut={stopDrawing}
          className="cursor-crosshair"
        />
        <canvas
          ref={maskCanvasRef}
          className="absolute inset-0 opacity-0 pointer-events-none"  // 숨겨진 마스크 캔버스
        />
      </div>

      <button
        onClick={handleSubmit}
        disabled={!image}
        className={`px-4 py-2 rounded ${
          image 
            ? 'bg-blue-500 text-white hover:bg-blue-600' 
            : 'bg-gray-300 text-gray-500'
        }`}
      >
        Edit Image
      </button>
    </div>
  );
};

export default ImageMaskEditor;