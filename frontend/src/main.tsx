import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// Debug information
console.log('App starting...', {
  NODE_ENV: import.meta.env.MODE,
  timestamp: new Date().toISOString()
});

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Root element not found');
}

try {
  const root = createRoot(rootElement);
  
  root.render(
    <StrictMode>
      <App />
    </StrictMode>
  );
  
  console.log('App rendered successfully');
} catch (error) {
  console.error('Failed to render app:', error);
  
  // Fallback error display
  rootElement.innerHTML = `
    <div style="
      min-height: 100vh;
      background-color: #fbfaf3;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
      font-family: system-ui, sans-serif;
    ">
      <div style="
        max-width: 400px;
        background: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
      ">
        <h2 style="color: #272426; margin-bottom: 16px;">Application Error</h2>
        <p style="color: #737373; margin-bottom: 20px;">
          The application failed to start. Please check the console for more details.
        </p>
        <button 
          onclick="window.location.reload()" 
          style="
            background: #37c2a3;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
          "
        >
          Reload Page
        </button>
      </div>
    </div>
  `;
}