import React from 'react';
import Layout from '@theme/Layout';
import RagChatbot from '@site/src/components/RagChatbot';
import ContentButtons from '@site/src/components/ContentButtons';

// Higher-order component to wrap pages with chatbot and content buttons
const withChatbotAndButtons = (WrappedComponent) => {
  return (props) => {
    const [currentContent, setCurrentContent] = React.useState('');
    
    // Function to update content when personalized or translated
    const handleContentUpdate = (newContent) => {
      setCurrentContent(newContent);
    };

    return (
      <Layout>
        <div style={{ position: 'relative' }}>
          <WrappedComponent {...props} updatedContent={currentContent} />
          <ContentButtons 
            content={props.updatedContent || props.children?.props?.content || ''} 
            onContentUpdate={handleContentUpdate} 
          />
          <RagChatbot mode="full" />
        </div>
      </Layout>
    );
  };
};

export default withChatbotAndButtons;