// @ts-check
/** @type {import('@docusaurus/types').PluginModule} */
module.exports = function chatbotPlugin(context, options) {
  return {
    name: 'chatbot-plugin',
    
    getClientModules() {
      return [
        require.resolve('./src/components/RagChatbot.js'),
        require.resolve('./src/components/ContentButtons.js')
      ];
    },
    
    // Add the chatbot to all pages
    injectHtmlTags() {
      return {
        postBodyTags: [
          `<div id="chatbot-container"></div>`,
        ],
      };
    },
  };
};