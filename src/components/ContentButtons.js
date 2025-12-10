import React, { useState } from 'react';
import { useLocation } from '@docusaurus/router';

const ContentButtons = ({ content, onContentUpdate }) => {
  const [isPersonalizing, setIsPersonalizing] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const location = useLocation();
  const moduleId = location.pathname.split('/').pop(); // Extract module ID from URL

  // Personalize content
  const handlePersonalize = async () => {
    setIsPersonalizing(true);
    try {
      // Get current user ID from localStorage (would be set after login)
      const userId = localStorage.getItem('userId');
      
      const response = await fetch(`/api/v1/personalize/content`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content: content,
          userId: userId || null,
          section: moduleId
        }),
      });
      
      const data = await response.json();
      if (data.personalized) {
        onContentUpdate(data.personalized);
      }
    } catch (error) {
      console.error('Error personalizing content:', error);
    } finally {
      setIsPersonalizing(false);
    }
  };

  // Translate to Urdu
  const handleTranslateToUrdu = async () => {
    setIsTranslating(true);
    try {
      // Get current user ID from localStorage (would be set after login)
      const userId = localStorage.getItem('userId');
      
      const response = await fetch(`/api/v1/translate/to-urdu`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: content,
          userId: userId || null
        }),
      });
      
      const data = await response.json();
      if (data.translated) {
        onContentUpdate(data.translated);
      }
    } catch (error) {
      console.error('Error translating to Urdu:', error);
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <div style={{ 
      display: 'flex', 
      gap: '10px', 
      marginBottom: '20px',
      flexWrap: 'wrap'
    }}>
      <button
        onClick={handlePersonalize}
        disabled={isPersonalizing}
        style={{
          padding: '8px 16px',
          backgroundColor: '#4F6FFF',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          cursor: isPersonalizing ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}
      >
        {isPersonalizing ? (
          <>
            <span className="loading-spinner">⏳</span> Personalizing...
          </>
        ) : (
          <>
            <span>🔄</span> Personalize
          </>
        )}
      </button>
      
      <button
        onClick={handleTranslateToUrdu}
        disabled={isTranslating}
        style={{
          padding: '8px 16px',
          backgroundColor: '#10B981',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          cursor: isTranslating ? 'not-allowed' : 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: '6px'
        }}
      >
        {isTranslating ? (
          <>
            <span className="loading-spinner">⏳</span> Translating...
          </>
        ) : (
          <>
            <span>🇵🇰</span> اردو میں دیکھیں
          </>
        )}
      </button>
    </div>
  );
};

export default ContentButtons;