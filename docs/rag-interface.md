---
title: RAG Chat Interface
---

# RAG-Powered Course Assistant

<div id="rag-chat-container">
  <div id="chat-messages" style={{ maxHeight: "400px", overflowY: "auto", border: "1px solid #ccc", padding: "10px", marginBottom: "10px" }}>
    <div>Welcome to the Physical AI & Humanoid Robotics course assistant! Ask any questions about the course content.</div>
  </div>
  <div style={{ display: "flex" }}>
    <input 
      type="text" 
      id="user-input" 
      placeholder="Ask a question about the course..." 
      style={{ flex: 1, padding: "8px" }}
    />
    <button id="send-btn" style={{ marginLeft: "10px", padding: "8px 15px" }}>Send</button>
  </div>
  <div style={{ marginTop: "10px" }}>
    <label>
      <input type="checkbox" id="urdu-toggle" /> Enable Urdu Support
    </label>
  </div>
</div>

<script>
  // Simple client-side implementation for demonstration
  document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const urduToggle = document.getElementById('urdu-toggle');
    
    // Function to add a message to the chat
    function addMessage(text, isUser = false) {
      const messageDiv = document.createElement('div');
      messageDiv.style.marginBottom = '10px';
      messageDiv.style.textAlign = isUser ? 'right' : 'left';
      messageDiv.style.padding = '8px';
      messageDiv.style.borderRadius = '4px';
      messageDiv.style.backgroundColor = isUser ? '#e3f2fd' : '#f5f5f5';
      messageDiv.innerHTML = text;
      chatMessages.appendChild(messageDiv);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Function to simulate RAG response
    async function getRagResponse(question, language = 'en') {
      // In a real implementation, this would call the backend API
      // For this example, we'll return a simulated response
      return `This is a simulated response for: "${question}". In the actual implementation, this would connect to the RAG system to provide contextually relevant answers based on the course content.`;
    }
    
    // Handle send button click
    sendBtn.addEventListener('click', async function() {
      const question = userInput.value.trim();
      if (!question) return;
      
      // Add user message to chat
      addMessage(`<strong>You:</strong> ${question}`, true);
      userInput.value = '';
      
      // Show loading message
      const loadingId = `loading-${Date.now()}`;
      addMessage(`<div id="${loadingId}">Processing...</div>`);
      
      // Get response from RAG system
      const language = urduToggle.checked ? 'ur' : 'en';
      const response = await getRagResponse(question, language);
      
      // Remove loading message and add actual response
      const loadingElement = document.getElementById(loadingId);
      if (loadingElement) {
        loadingElement.parentElement.innerHTML = `<strong>Assistant:</strong> ${response}`;
      }
    });
    
    // Allow sending with Enter key
    userInput.addEventListener('keypress', function(e) {
      if (e.key === 'Enter') {
        sendBtn.click();
      }
    });
  });
</script>

## How It Works

This RAG (Retrieval-Augmented Generation) interface allows you to ask questions about the course materials and get contextually relevant responses based on the content of the Physical AI & Humanoid Robotics course. The system:

1. Takes your question as input
2. Searches through the course documents to find relevant information
3. Generates a response based on the retrieved information
4. Provides citations to the source materials

## Features

- **Course Content Search**: Ask questions about any topic covered in the course
- **Multi-language Support**: Toggle between English and Urdu
- **Source Attribution**: Responses include references to the course materials
- **Contextual Understanding**: The system understands context and provides relevant answers