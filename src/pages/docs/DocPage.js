import React from 'react';
import { MDXProvider } from '@mdx-js/react';
import { useMDXComponents } from '@theme/init';
import MDXComponents from '@theme/MDXComponents';
import Layout from '@theme/Layout';
import ContentButtons from '@site/src/components/ContentButtons';
import RagChatbot from '@site/src/components/RagChatbot';

// Custom DocPage component that adds content buttons
function DocPage(props) {
  const { content: DocContent } = props;
  const { metadata, frontMatter } = DocContent;
  
  const [updatedContent, setUpdatedContent] = React.useState(null);
  
  return (
    <Layout
      title={metadata.title}
      description={metadata.description}
      wrapperClassName="docs-wrapper"
      searchMetadata={{ version: metadata.version }}
    >
      <div style={{ position: 'relative' }}>
        <div className="container padding-top--md padding-bottom--lg">
          <div className="row">
            <main className="col col--9 col--offset-1">
              <div className="margin-bottom--lg">
                <ContentButtons 
                  content={updatedContent || DocContent.content || ''} 
                  onContentUpdate={setUpdatedContent} 
                />
              </div>
              
              {updatedContent ? (
                <div className="markdown">
                  {updatedContent}
                </div>
              ) : (
                <MDXProvider components={useMDXComponents(MDXComponents)}>
                  <DocContent />
                </MDXProvider>
              )}
            </main>
          </div>
        </div>
        
        <RagChatbot mode="full" />
      </div>
    </Layout>
  );
}

export default DocPage;