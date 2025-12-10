import React, { useState } from 'react';
import Layout from '@theme/Layout';
import { useHistory } from '@docusaurus/router';

function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [isSignup, setIsSignup] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const history = useHistory();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      if (isSignup) {
        // Sign up flow
        const signupResponse = await fetch('/api/v1/auth/signup', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ email, name }),
        });

        if (!signupResponse.ok) {
          const errorData = await signupResponse.json();
          throw new Error(errorData.detail || 'Signup failed');
        }

        const userData = await signupResponse.json();
        
        // Save user ID to localStorage
        localStorage.setItem('userId', userData.id);
        
        // Redirect to background questions page
        history.push('/auth/background');
      } else {
        // Login flow - for now just store email in localStorage
        localStorage.setItem('userEmail', email);
        history.push('/');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Layout title="Authentication" description="Login or sign up to personalize your experience">
      <div style={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '70vh',
        padding: '20px'
      }}>
        <div style={{
          width: '100%',
          maxWidth: '400px',
          backgroundColor: 'var(--ifm-color-emphasis-100)',
          padding: '30px',
          borderRadius: '8px',
          boxShadow: '0 4px 14px 0 rgba(0,0,0,0.1)'
        }}>
          <h2 style={{ textAlign: 'center', marginBottom: '24px' }}>
            {isSignup ? 'Create Account' : 'Login to Your Account'}
          </h2>
          
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
          
          <form onSubmit={handleSubmit}>
            {isSignup && (
              <div style={{ marginBottom: '16px' }}>
                <label htmlFor="name" style={{ display: 'block', marginBottom: '6px' }}>
                  Full Name
                </label>
                <input
                  id="name"
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required={isSignup}
                  style={{
                    width: '100%',
                    padding: '10px',
                    border: '1px solid var(--ifm-color-emphasis-300)',
                    borderRadius: '4px',
                    fontSize: '16px'
                  }}
                />
              </div>
            )}
            
            <div style={{ marginBottom: '16px' }}>
              <label htmlFor="email" style={{ display: 'block', marginBottom: '6px' }}>
                Email Address
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                style={{
                  width: '100%',
                  padding: '10px',
                  border: '1px solid var(--ifm-color-emphasis-300)',
                  borderRadius: '4px',
                  fontSize: '16px'
                }}
              />
            </div>
            
            {isSignup && (
              <div style={{ marginBottom: '24px' }}>
                <label htmlFor="password" style={{ display: 'block', marginBottom: '6px' }}>
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required={isSignup}
                  style={{
                    width: '100%',
                    padding: '10px',
                    border: '1px solid var(--ifm-color-emphasis-300)',
                    borderRadius: '4px',
                    fontSize: '16px'
                  }}
                />
              </div>
            )}
            
            <button
              type="submit"
              disabled={isLoading}
              style={{
                width: '100%',
                padding: '12px',
                backgroundColor: '#4F6FFF',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                fontSize: '16px',
                cursor: isLoading ? 'not-allowed' : 'pointer'
              }}
            >
              {isLoading ? 'Processing...' : (isSignup ? 'Sign Up' : 'Login')}
            </button>
          </form>
          
          <div style={{ textAlign: 'center', marginTop: '16px' }}>
            <button
              onClick={() => setIsSignup(!isSignup)}
              style={{
                background: 'none',
                border: 'none',
                color: '#4F6FFF',
                cursor: 'pointer',
                fontSize: '14px'
              }}
            >
              {isSignup 
                ? 'Already have an account? Login' 
                : "Don't have an account? Sign up"}
            </button>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default LoginPage;