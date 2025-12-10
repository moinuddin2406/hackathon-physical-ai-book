import React, { useState, useEffect, useRef } from 'react';
import { useColorMode } from '@docusaurus/theme-common';

// Chatbot component that can be embedded in Docusaurus pages
const RagChatbot = ({ mode = 'full' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedText, setSelectedText] = useState('');
  const messagesEndRef = useRef(null);
  const { colorMode } = useColorMode();

  // Get selected text from the page
  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      if (selection.toString().trim() !== '') {
        setSelectedText(selection.toString().trim());
      }
    };

    document.addEventListener('mouseup', handleSelection);
    return () => {
      document.removeEventListener('mouseup', handleSelection);
    };
  }, []);

  // Scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    // Add user message
    const userMessage = { role: 'user', content: inputValue };
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Call the backend API
      const response = await fetch('/api/v1/rag/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: inputValue,
          language: 'en',
        }),
      });

      const data = await response.json();
      
      // Add bot response
      const botMessage = { 
        role: 'assistant', 
        content: data.response,
        sources: data.sourceDocuments || []
      };
      
      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error fetching response:', error);
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const askAboutSelected = () => {
    if (!selectedText) return;
    
    // Add user message about selected text
    const userMessage = { role: 'user', content: `Explain this: ${selectedText}` };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    // Call the backend API
    fetch('/api/v1/rag/query', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question: `Explain this: ${selectedText}`,
        language: 'en',
      }),
    })
    .then(response => response.json())
    .then(data => {
      const botMessage = { 
        role: 'assistant', 
        content: data.response,
        sources: data.sourceDocuments || []
      };
      
      setMessages(prev => [...prev, botMessage]);
    })
    .catch(error => {
      console.error('Error fetching response:', error);
      const errorMessage = { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      };
      setMessages(prev => [...prev, errorMessage]);
    })
    .finally(() => {
      setIsLoading(false);
    });
  };

  // Toggle chat window
  const toggleChat = () => {
    setIsOpen(!isOpen);
  };

  // Clear chat
  const clearChat = () => {
    setMessages([]);
  };

  return (
    <div className="rag-chatbot">
      {/* Chatbot button */}
      {!isOpen && (
        <button
          onClick={toggleChat}
          className="chatbot-button"
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '60px',
            height: '60px',
            borderRadius: '50%',
            backgroundColor: '#4F6FFF',
            color: 'white',
            border: 'none',
            fontSize: '24px',
            cursor: 'pointer',
            zIndex: 1000,
            boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
          }}
        >
          💬
        </button>
      )}

      {/* Chatbot window */}
      {isOpen && (
        <div 
          className="chatbot-window"
          style={{
            position: 'fixed',
            bottom: '20px',
            right: '20px',
            width: '380px',
            height: mode === 'full' ? '500px' : '400px',
            backgroundColor: colorMode === 'dark' ? '#1a1a1a' : '#ffffff',
            border: '1px solid #ddd',
            borderRadius: '12px',
            display: 'flex',
            flexDirection: 'column',
            zIndex: 1000,
            boxShadow: '0 8px 30px rgba(0,0,0,0.12)',
            overflow: 'hidden',
          }}
        >
          {/* Header */}
          <div 
            style={{
              padding: '16px',
              backgroundColor: '#4F6FFF',
              color: 'white',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <h3 style={{ margin: 0, fontSize: '16px' }}>
              {mode === 'full' ? 'Course Assistant' : 'Selected Text Assistant'}
            </h3>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                onClick={clearChat}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '16px',
                }}
              >
                🗑️
              </button>
              <button 
                onClick={toggleChat}
                style={{
                  background: 'none',
                  border: 'none',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '16px',
                }}
              >
                ×
              </button>
            </div>
          </div>

          {/* Messages container */}
          <div 
            style={{
              flex: 1,
              padding: '16px',
              overflowY: 'auto',
              backgroundColor: colorMode === 'dark' ? '#111827' : '#f9fafb',
            }}
          >
            {messages.length === 0 && mode === 'full' && (
              <div style={{ textAlign: 'center', marginTop: '40px', color: colorMode === 'dark' ? '#9CA3AF' : '#6B7280' }}>
                <p>Ask me anything about the course content!</p>
              </div>
            )}
            {messages.length === 0 && mode === 'selected' && selectedText && (
              <div>
                <p style={{ color: colorMode === 'dark' ? '#D1D5DB' : '#4B5563' }}>
                  <strong>Selected text:</strong> {selectedText}
                </p>
                <button 
                  onClick={askAboutSelected}
                  disabled={isLoading || !selectedText}
                  style={{
                    marginTop: '10px',
                    padding: '8px 16px',
                    backgroundColor: selectedText ? '#4F6FFF' : '#9CA3AF',
                    color: 'white',
                    border: 'none',
                    borderRadius: '6px',
                    cursor: selectedText ? 'pointer' : 'not-allowed',
                  }}
                >
                  {isLoading ? 'Asking...' : 'Ask about this'}
                </button>
              </div>
            )}
            {messages.map((msg, index) => (
              <div 
                key={index} 
                style={{
                  marginBottom: '16px',
                  textAlign: msg.role === 'user' ? 'right' : 'left',
                }}
              >
                <div
                  style={{
                    display: 'inline-block',
                    padding: '10px 14px',
                    borderRadius: '18px',
                    backgroundColor: msg.role === 'user' 
                      ? (colorMode === 'dark' ? '#374151' : '#E5E7EB') 
                      : (colorMode === 'dark' ? '#312E81' : '#E0E7FF'),
                    color: msg.role === 'user' 
                      ? (colorMode === 'dark' ? '#F3F4F6' : '#1F2937') 
                      : (colorMode === 'dark' ? '#E0E7FF' : '#1E3A8A'),
                    maxWidth: '85%',
                  }}
                >
                  {msg.content}
                  {msg.sources && msg.sources.length > 0 && (
                    <div style={{ marginTop: '8px', fontSize: '0.8em', opacity: 0.8 }}>
                      Sources: {msg.sources.map(s => s.sectionTitle || s.moduleId).join(', ')}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div style={{ textAlign: 'left', marginBottom: '16px' }}>
                <div
                  style={{
                    display: 'inline-block',
                    padding: '10px 14px',
                    borderRadius: '18px',
                    backgroundColor: colorMode === 'dark' ? '#312E81' : '#E0E7FF',
                    color: colorMode === 'dark' ? '#E0E7FF' : '#1E3A8A',
                  }}
                >
                  Thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <form
            onSubmit={handleSubmit}
            style={{
              padding: '12px',
              backgroundColor: colorMode === 'dark' ? '#1F2937' : '#ffffff',
              borderTop: `1px solid ${colorMode === 'dark' ? '#374151' : '#E5E7EB'}`,
              display: 'flex',
              gap: '8px',
            }}
          >
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask about the course..."
              style={{
                flex: 1,
                padding: '10px 12px',
                borderRadius: '24px',
                border: `1px solid ${colorMode === 'dark' ? '#4B5563' : '#D1D5DB'}`,
                backgroundColor: colorMode === 'dark' ? '#111827' : '#ffffff',
                color: colorMode === 'dark' ? '#F3F4F6' : '#1F2937',
              }}
              disabled={isLoading}
            />
            <button
              type="submit"
              style={{
                padding: '10px 16px',
                backgroundColor: '#4F6FFF',
                color: 'white',
                border: 'none',
                borderRadius: '24px',
                cursor: isLoading ? 'not-allowed' : 'pointer',
                opacity: isLoading ? 0.6 : 1,
              }}
              disabled={isLoading || !inputValue.trim()}
            >
              Send
            </button>
          </form>
        </div>
      )}
    </div>
  );
};

export default RagChatbot;