import React, { useState } from 'react';
import Layout from '@theme/Layout';
import { useHistory } from '@docusaurus/router';

function BackgroundQuestions() {
  const [currentStep, setCurrentStep] = useState(0);
  const [backgroundInfo, setBackgroundInfo] = useState({
    programming_experience: '',
    robotics_experience: '',
    education_level: '',
    primary_language: 'en',
    learning_goals: ''
  });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');

  const history = useHistory();

  const questions = [
    {
      key: 'programming_experience',
      question: 'What is your programming experience level?',
      options: [
        { value: 'beginner', label: 'Beginner (just starting)' },
        { value: 'intermediate', label: 'Intermediate (some experience)' },
        { value: 'advanced', label: 'Advanced (extensive experience)' }
      ]
    },
    {
      key: 'robotics_experience',
      question: 'What is your robotics experience?',
      options: [
        { value: 'none', label: 'No experience' },
        { value: 'basic', label: 'Basic experience' },
        { value: 'intermediate', label: 'Intermediate experience' },
        { value: 'advanced', label: 'Advanced experience' }
      ]
    },
    {
      key: 'education_level',
      question: 'What is your highest education level?',
      options: [
        { value: 'high_school', label: 'High School' },
        { value: 'undergraduate', label: 'Undergraduate' },
        { value: 'graduate', label: 'Graduate' },
        { value: 'professional', label: 'Professional/Industry' }
      ]
    },
    {
      key: 'primary_language',
      question: 'What is your primary language?',
      options: [
        { value: 'en', label: 'English' },
        { value: 'ur', label: 'Urdu' },
        { value: 'other', label: 'Other' }
      ]
    },
    {
      key: 'learning_goals',
      question: 'What are your learning goals?',
      type: 'textarea',
      placeholder: 'Tell us what you hope to achieve from this course...'
    }
  ];

  const handleOptionChange = (key, value) => {
    setBackgroundInfo(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleNext = () => {
    if (currentStep < questions.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      handleSubmit();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    setError('');

    try {
      // Get user ID from localStorage
      const userId = localStorage.getItem('userId');
      
      if (!userId) {
        throw new Error('No user ID found. Please sign up first.');
      }

      const response = await fetch(`/api/v1/auth/background/${userId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(backgroundInfo),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to save background information');
      }

      // Redirect to home page
      history.push('/');
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const currentQuestion = questions[currentStep];
  const progress = ((currentStep + 1) / questions.length) * 100;

  return (
    <Layout title="Background Information" description="Tell us about yourself to personalize your learning experience">
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '70vh',
        padding: '20px'
      }}>
        <div style={{
          width: '100%',
          maxWidth: '600px',
          backgroundColor: 'var(--ifm-color-emphasis-100)',
          padding: '30px',
          borderRadius: '8px',
          boxShadow: '0 4px 14px 0 rgba(0,0,0,0.1)'
        }}>
          <h2 style={{ textAlign: 'center', marginBottom: '10px' }}>
            Personalize Your Experience
          </h2>
          <p style={{ textAlign: 'center', color: 'var(--ifm-color-emphasis-700)', marginBottom: '24px' }}>
            Help us tailor the content to your background
          </p>
          
          {/* Progress bar */}
          <div style={{
            width: '100%',
            height: '8px',
            backgroundColor: 'var(--ifm-color-emphasis-200)',
            borderRadius: '4px',
            marginBottom: '24px',
            overflow: 'hidden'
          }}>
            <div 
              style={{
                width: `${progress}%`,
                height: '100%',
                backgroundColor: '#4F6FFF',
                transition: 'width 0.3s ease'
              }}
            />
          </div>
          
          {error && (
            <div style={{
              color: '#e3342f',
              backgroundColor: '#fcebea',
              padding: '10px',
              borderRadius: '4px',
              marginBottom: '16px'
            }}>
              {error}
            </div>
          )}
          
          <div style={{ marginBottom: '24px' }}>
            <h3 style={{ marginBottom: '16px' }}>
              {currentQuestion.question}
            </h3>
            
            {currentQuestion.type === 'textarea' ? (
              <textarea
                value={backgroundInfo[currentQuestion.key]}
                onChange={(e) => handleOptionChange(currentQuestion.key, e.target.value)}
                placeholder={currentQuestion.placeholder}
                rows={4}
                style={{
                  width: '100%',
                  padding: '12px',
                  border: '1px solid var(--ifm-color-emphasis-300)',
                  borderRadius: '4px',
                  fontSize: '16px',
                  resize: 'vertical'
                }}
              />
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {currentQuestion.options.map((option) => (
                  <label 
                    key={option.value}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      padding: '12px',
                      border: `2px solid ${backgroundInfo[currentQuestion.key] === option.value ? '#4F6FFF' : 'var(--ifm-color-emphasis-200)'}`,
                      borderRadius: '6px',
                      cursor: 'pointer',
                      transition: 'border-color 0.2s'
                    }}
                  >
                    <input
                      type="radio"
                      name={currentQuestion.key}
                      value={option.value}
                      checked={backgroundInfo[currentQuestion.key] === option.value}
                      onChange={() => handleOptionChange(currentQuestion.key, option.value)}
                      style={{ marginRight: '12px' }}
                    />
                    <span>{option.label}</span>
                  </label>
                ))}
              </div>
            )}
          </div>
          
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <button
              onClick={handlePrevious}
              disabled={currentStep === 0}
              style={{
                padding: '10px 20px',
                backgroundColor: currentStep === 0 ? 'var(--ifm-color-emphasis-200)' : 'var(--ifm-color-emphasis-300)',
                border: 'none',
                borderRadius: '4px',
                cursor: currentStep === 0 ? 'not-allowed' : 'pointer'
              }}
            >
              Previous
            </button>
            
            <button
              onClick={handleNext}
              disabled={currentStep < questions.length - 1 && !backgroundInfo[currentQuestion.key]}
              style={{
                padding: '10px 20px',
                backgroundColor: (currentStep < questions.length - 1 && !backgroundInfo[currentQuestion.key]) 
                  ? 'var(--ifm-color-emphasis-200)' 
                  : '#4F6FFF',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: (currentStep < questions.length - 1 && !backgroundInfo[currentQuestion.key]) 
                  ? 'not-allowed' 
                  : 'pointer'
              }}
            >
              {currentStep === questions.length - 1 ? 'Finish' : 'Next'}
            </button>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default BackgroundQuestions;