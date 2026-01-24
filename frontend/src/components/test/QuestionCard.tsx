import React, { useState, useRef } from "react";
import { Question } from "../../types";
import { MathText } from "../ui/MathRenderer";
import { Card, CardContent } from "../ui/Card";
import { Button } from "../ui/Button";
import { Calculator, Image as ImageIcon, X } from "lucide-react";
import { ImageCropper } from "./ImageCropper";

interface QuestionCardProps {
  question: Question;
  answer: any;
  onAnswerChange: (value: any) => void;
}

export function QuestionCard({
  question,
  answer,
  onAnswerChange,
}: QuestionCardProps) {
  const [showCropper, setShowCropper] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isMCQ =
    question.question_type === "mcq_single" ||
    question.question_type === "mcq_multi";
  const isTrueFalse = question.question_type === "true_false";
  const options = isTrueFalse ? ["True", "False"] : question.options;

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedImage(e.target.files[0]);
      setShowCropper(true);
    }
  };

  const handleCropComplete = (croppedBlob: Blob) => {
    const url = URL.createObjectURL(croppedBlob);
    onAnswerChange({
      ...answer,
      image: {
        blob: croppedBlob,
        previewUrl: url,
      },
    });
    setShowCropper(false);
    setSelectedImage(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleRemoveImage = () => {
    const newAnswer = { ...answer };
    delete newAnswer.image;
    onAnswerChange(newAnswer);
  };

  const handleTextChange = (text: string) => {
    onAnswerChange({
      ...answer,
      text: text,
    });
  };

  const handleMCQSelect = (optionIndex: number) => {
    if (question.question_type === "mcq_multi") {
      const currentSelected = (answer?.selectedOptions as number[]) || [];
      const newSelected = currentSelected.includes(optionIndex)
        ? currentSelected.filter((i) => i !== optionIndex)
        : [...currentSelected, optionIndex];
      onAnswerChange({ ...answer, selectedOptions: newSelected });
    } else {
      // Single select (works for True/False too)
      onAnswerChange({ ...answer, selectedOptions: [optionIndex] });
    }
  };

  return (
    <Card className="mb-6 overflow-visible">
      <CardContent className="p-6">
        <div className="flex justify-between items-start mb-4">
          <h3 className="font-semibold text-dark flex items-center text-lg">
            Q{question.question_number}
            {question.contains_math_expression && (
              <Calculator className="h-4 w-4 ml-2 text-blue-600" />
            )}
          </h3>
          <div className="flex items-center space-x-2 text-sm">
            <span className="text-secondary-800 bg-secondary-50 px-2 py-1 rounded font-medium">
              {question.question_type.replace("_", " ").toUpperCase()}
            </span>
            <span className="text-green-800 bg-green-50 px-2 py-1 rounded font-medium">
              {question.maximum_marks} Marks
            </span>
          </div>
        </div>

        <div className="text-neutral-800 mb-6 text-lg">
          {question.contains_math_expression ? (
            <MathText
              text={question.question_text}
              className="leading-relaxed"
            />
          ) : (
            <p>{question.question_text}</p>
          )}
        </div>

        {isMCQ || isTrueFalse ? (
          <div className="space-y-3">
            {options?.map((option, index) => {
              const isSelected = (
                answer?.selectedOptions as number[]
              )?.includes(index);
              return (
                <div
                  key={index}
                  onClick={() => handleMCQSelect(index)}
                  className={`
                    flex items-center p-4 rounded-lg border-2 cursor-pointer transition-all
                    ${
                      isSelected
                        ? "border-primary-500 bg-primary-50"
                        : "border-neutral-200 hover:border-primary-200 hover:bg-neutral-50"
                    }
                  `}
                >
                  <div
                    className={`
                    w-6 h-6 rounded-full border-2 flex items-center justify-center mr-4 flex-shrink-0
                    ${
                      isSelected
                        ? "border-primary-600 bg-primary-600 text-white"
                        : "border-neutral-400"
                    }
                  `}
                  >
                    {isSelected && (
                      <div className="w-2 h-2 bg-white rounded-full" />
                    )}
                  </div>
                  <div className="flex-1">
                    {question.contains_math_expression ? (
                      <MathText
                        text={typeof option === "string" ? option : option.text}
                      />
                    ) : (
                      <span>
                        {typeof option === "string" ? option : option.text}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="space-y-4">
            <div className="relative">
              <textarea
                value={answer?.text || ""}
                onChange={(e) => handleTextChange(e.target.value)}
                placeholder="Type your answer here..."
                className="w-full min-h-[150px] p-4 rounded-lg border border-neutral-300 focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-y"
              />
            </div>

            <div className="flex items-center gap-4">
              <input
                type="file"
                accept="image/*"
                onChange={handleImageSelect}
                className="hidden"
                ref={fileInputRef}
              />

              {!answer?.image ? (
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => fileInputRef.current?.click()}
                  className="flex items-center"
                >
                  <ImageIcon className="h-4 w-4 mr-2" />
                  Add Image Answer
                </Button>
              ) : (
                <div className="relative inline-block">
                  <img
                    src={answer.image.previewUrl}
                    alt="Answer"
                    className="h-32 w-auto rounded-lg border border-neutral-200 object-cover"
                  />
                  <button
                    onClick={handleRemoveImage}
                    className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 hover:bg-red-600 shadow-sm"
                  >
                    <X className="h-3 w-3" />
                  </button>
                </div>
              )}
            </div>
          </div>
        )}
      </CardContent>

      {showCropper && selectedImage && (
        <ImageCropper
          imageFile={selectedImage}
          onCropComplete={handleCropComplete}
          onCancel={() => {
            setShowCropper(false);
            setSelectedImage(null);
            if (fileInputRef.current) fileInputRef.current.value = "";
          }}
        />
      )}
    </Card>
  );
}
