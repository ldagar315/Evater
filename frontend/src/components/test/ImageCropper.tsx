import React, { useState, useRef, useEffect } from "react";
import { X, Check, RefreshCw } from "lucide-react";
import { Button } from "../ui/Button";
import { Point, warpImage } from "../../utils/imageProcessing";

interface ImageCropperProps {
  imageFile: File;
  onCropComplete: (croppedImage: Blob) => void;
  onCancel: () => void;
}

export function ImageCropper({
  imageFile,
  onCropComplete,
  onCancel,
}: ImageCropperProps) {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [corners, setCorners] = useState<Point[]>([]);
  const [activeCornerIndex, setActiveCornerIndex] = useState<number | null>(
    null
  );
  const [isProcessing, setIsProcessing] = useState(false);
  const imageRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setImageSrc(e.target?.result as string);
    };
    reader.readAsDataURL(imageFile);
  }, [imageFile]);

  // Initialize corners when image loads
  const handleImageLoad = () => {
    if (imageRef.current) {
      const { width, height } = imageRef.current;
      const padding = 20; // px
      // Default corners: 10% inset or fixed padding
      setCorners([
        { x: padding, y: padding }, // TL
        { x: width - padding, y: padding }, // TR
        { x: width - padding, y: height - padding }, // BR
        { x: padding, y: height - padding }, // BL
      ]);
    }
  };

  const getScaledCoordinates = (e: React.MouseEvent | React.TouchEvent) => {
    if (!imageRef.current) return null;
    const rect = imageRef.current.getBoundingClientRect();

    // Handle both mouse and touch events
    const clientX =
      "touches" in e ? e.touches[0].clientX : (e as React.MouseEvent).clientX;
    const clientY =
      "touches" in e ? e.touches[0].clientY : (e as React.MouseEvent).clientY;

    const x = clientX - rect.left;
    const y = clientY - rect.top;
    return { x, y };
  };

  const handleMouseDown = (index: number) => {
    setActiveCornerIndex(index);
  };

  const handleMouseMove = (e: React.MouseEvent | React.TouchEvent) => {
    if (activeCornerIndex === null || !imageRef.current) return;

    const coords = getScaledCoordinates(e);
    if (!coords) return;

    // Constrain to image bounds
    const x = Math.max(0, Math.min(coords.x, imageRef.current.width));
    const y = Math.max(0, Math.min(coords.y, imageRef.current.height));

    setCorners((prev) => {
      const newCorners = [...prev];
      newCorners[activeCornerIndex] = { x, y };
      return newCorners;
    });
  };

  const handleMouseUp = () => {
    setActiveCornerIndex(null);
  };

  const handleCrop = async () => {
    if (!imageRef.current || corners.length !== 4) return;
    setIsProcessing(true);

    try {
      // Map UI coordinates to natural image coordinates
      const scaleX = imageRef.current.naturalWidth / imageRef.current.width;
      const scaleY = imageRef.current.naturalHeight / imageRef.current.height;

      const naturalCorners = corners.map((p) => ({
        x: p.x * scaleX,
        y: p.y * scaleY,
      }));

      const blob = await warpImage(imageRef.current, naturalCorners);
      onCropComplete(blob);
    } catch (error) {
      console.error("Crop failed:", error);
      // Optionally show error to user
    } finally {
      setIsProcessing(false);
    }
  };

  const resetCorners = () => {
    if (imageRef.current) {
      const { width, height } = imageRef.current;
      const padding = 20;
      setCorners([
        { x: padding, y: padding },
        { x: width - padding, y: padding },
        { x: width - padding, y: height - padding },
        { x: padding, y: height - padding },
      ]);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-85 p-4 backdrop-blur-sm">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-4xl overflow-hidden flex flex-col max-h-[90vh]">
        <div className="p-4 border-b flex justify-between items-center bg-white z-10">
          <div>
            <h3 className="font-bold text-lg text-gray-900">
              Adjust Document Corners
            </h3>
            <p className="text-sm text-gray-500">
              Drag the corners to match the document edges
            </p>
          </div>
          <button
            onClick={onCancel}
            className="text-gray-400 hover:text-gray-600 transition-colors p-2 hover:bg-gray-100 rounded-full"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        <div
          className="flex-1 bg-gray-900 overflow-auto relative flex items-center justify-center p-8"
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchMove={handleMouseMove}
          onTouchEnd={handleMouseUp}
          onTouchCancel={handleMouseUp}
        >
          {imageSrc && (
            <div className="relative shadow-2xl inline-block">
              <img
                ref={imageRef}
                src={imageSrc}
                alt="Crop target"
                className="max-w-full max-h-[60vh] object-contain select-none pointer-events-none"
                onLoad={handleImageLoad}
                draggable={false}
              />

              {/* SVG Overlay */}
              <svg
                className="absolute inset-0 w-full h-full pointer-events-none"
                style={{ zIndex: 10 }}
              >
                {/* Connecting Lines */}
                <polygon
                  points={corners.map((p) => `${p.x},${p.y}`).join(" ")}
                  fill="rgba(59, 130, 246, 0.2)"
                  stroke="#3b82f6"
                  strokeWidth="2"
                  vectorEffect="non-scaling-stroke"
                />

                {/* Corner Handles */}
                {corners.map((point, index) => (
                  <circle
                    key={index}
                    cx={point.x}
                    cy={point.y}
                    r={8}
                    className="fill-blue-500 stroke-white stroke-2 cursor-move pointer-events-auto hover:scale-125 transition-transform"
                    onMouseDown={(e) => {
                      e.stopPropagation();
                      handleMouseDown(index);
                    }}
                    onTouchStart={(e) => {
                      e.stopPropagation();
                      handleMouseDown(index);
                    }}
                  />
                ))}
              </svg>
            </div>
          )}
        </div>

        <div className="p-4 bg-gray-50 border-t flex justify-between items-center">
          <Button
            variant="outline"
            onClick={resetCorners}
            className="flex items-center text-gray-600"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Reset Corners
          </Button>

          <div className="flex space-x-3">
            <Button variant="ghost" onClick={onCancel}>
              Cancel
            </Button>
            <Button
              onClick={handleCrop}
              className="flex items-center min-w-[100px]"
              disabled={isProcessing}
            >
              {isProcessing ? (
                <span className="animate-pulse">Processing...</span>
              ) : (
                <>
                  <Check className="w-4 h-4 mr-2" />
                  Crop & Save
                </>
              )}
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
