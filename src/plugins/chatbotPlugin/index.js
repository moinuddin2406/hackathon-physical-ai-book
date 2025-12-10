// @ts-check

/** @type {import('@docusaurus/types').PluginModule} */
const chatbotPlugin = (context, options) => {
  return {
    name: 'chatbot-plugin',

    getClientModules() {
      return [
        require.resolve('./../../components/RagChatbot'),
        require.resolve('./../../components/ContentButtons')
      ];
    },

    // Add the chatbot to all pages
    injectHtmlTags() {
      return {
        postBodyTags: [
          '<div id="chatbot-container"></div>',
        ],
      };
    },
  };
};

module.exports = chatbotPlugin; 