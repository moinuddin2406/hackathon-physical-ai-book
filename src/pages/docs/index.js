import React from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';

function DocsHome() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Docs - ${siteConfig.title}`}
      description="Documentation for the Physical AI & Humanoid Robotics course">
      <main style={{ padding: '2rem' }}>
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          justifyContent: 'center',
          minHeight: '70vh',
          textAlign: 'center'
        }}>
          <h1 style={{ fontSize: '3rem', marginBottom: '1rem' }}>Course Documentation</h1>
          <p style={{ fontSize: '1.2rem', maxWidth: '600px' }}>
            Welcome to the documentation for {siteConfig.title}
          </p>
          <div style={{ marginTop: '2rem' }}>
            <Link 
              to="/docs/module1/robotic-nervous-system-ros2"
              style={{
                padding: '12px 24px',
                backgroundColor: '#4F6FFF',
                color: 'white',
                textDecoration: 'none',
                borderRadius: '6px',
                fontSize: '1.1rem'
              }}
            >
              Start with Module 1
            </Link>
          </div>
        </div>
      </main>
    </Layout>
  );
}

export default DocsHome;