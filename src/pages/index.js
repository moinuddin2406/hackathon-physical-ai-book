import React from 'react';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import RagChatbot from '../components/RagChatbot';

function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Physical AI & Humanoid Robotics Comprehensive Course">
      <main style={{ padding: '2rem' }}>
        <div style={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center', 
          justifyContent: 'center',
          minHeight: '70vh',
          textAlign: 'center'
        }}>
          <h1 style={{ fontSize: '3rem', marginBottom: '1rem' }}>{siteConfig.title}</h1>
          <p style={{ fontSize: '1.2rem', maxWidth: '600px' }}>
            {siteConfig.tagline}
          </p>
          <div style={{ marginTop: '2rem' }}>
            <p>Explore the documentation using the sidebar or start chatting with our AI assistant!</p>
          </div>
        </div>
      </main>
      <RagChatbot mode="full" />
    </Layout>
  );
}

export default Home;