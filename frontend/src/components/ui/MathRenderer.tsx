import React, { useEffect, useRef } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'

interface MathRendererProps {
  math: string
  displayMode?: boolean
  className?: string
}

export function MathRenderer({ math, displayMode = false, className = '' }: MathRendererProps) {
  const mathRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    if (mathRef.current) {
      try {
        katex.render(math, mathRef.current, {
          displayMode,
          throwOnError: false,
          errorColor: '#cc0000',
          strict: false,
          trust: false,
          macros: {
            "\\f": "#1f(#2)",
          }
        })
      } catch (error) {
        // If KaTeX fails, just show the raw math
        if (mathRef.current) {
          mathRef.current.textContent = math
        }
      }
    }
  }, [math, displayMode])

  return <span ref={mathRef} className={className} />
}

interface MathTextProps {
  text: string
  className?: string
}

export function MathText({ text, className = '' }: MathTextProps) {
  // Simple regex to detect LaTeX math expressions
  const mathRegex = /\$\$([\s\S]+?)\$\$|\$([\s\S]+?)\$|\\\(([\s\S]+?)\\\)|\\\[([\s\S]+?)\\\]/g;
  
  const parts = []
  let lastIndex = 0
  let match

  while ((match = mathRegex.exec(text)) !== null) {
    // Add text before the math
    if (match.index > lastIndex) {
      parts.push(
        <span key={`text-${lastIndex}`}>
          {text.slice(lastIndex, match.index)}
        </span>
      )
    }

    // Add the math part
    const mathContent = match[1] || match[2] || match[3] || match[4]; // $$ or $ delimited
    const isDisplayMode = !!match[1] || !!match[4]; // $$ is display mode
    
    parts.push(
      <MathRenderer
        key={`math-${match.index}`}
        math={mathContent}
        displayMode={isDisplayMode}
        className="math-expression"
      />
    )

    lastIndex = match.index + match[0].length
  }

  // Add remaining text
  if (lastIndex < text.length) {
    parts.push(
      <span key={`text-${lastIndex}`}>
        {text.slice(lastIndex)}
      </span>
    )
  }

  // If no math found, return original text
  if (parts.length === 0) {
    return <span className={className}>{text}</span>
  }

  return <span className={className}>{parts}</span>
}