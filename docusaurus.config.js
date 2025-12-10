// @ts-check
// `@type` JSDoc annotations allow TypeScript to infer types for your configuration
/** @type {import('@docusaurus/types').DocusaurusConfig} */
(module.exports = {
  title: 'Physical AI & Humanoid Robotics',
  tagline: 'Comprehensive Course on ROS 2, Gazebo, NVIDIA Isaac, and Vision-Language-Action Systems',
  url: 'https://moinuddin247.github.io', // Updated to your GitHub username
  baseUrl: '/hackathon-physical-ai-book/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  organizationName: 'moinuddin247', // Updated to your GitHub username
  projectName: 'hackathon-physical-ai-book', // Updated to your repository name
  trailingSlash: false,

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          // Please change this to your repo.
          editUrl: 'https://github.com/moinuddin247/hackathon-physical-ai-book/edit/main/',
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],

  plugins: [
    [
      require.resolve('./src/plugins/chatbotPlugin'),
      {
        // Chatbot plugin options
      }
    ]
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Physical AI & Robotics',
        logo: {
          alt: 'Robotics Logo',
          src: 'img/logo.svg',
        },
        items: [
          {
            type: 'doc',
            docId: 'module1/robotic-nervous-system-ros2',
            position: 'left',
            label: 'Course Modules',
          },
          {
            href: 'https://github.com/moinuddin247/hackathon-physical-ai-book',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Modules',
            items: [
              {
                label: 'ROS 2 Fundamentals',
                to: '/docs/module1/robotic-nervous-system-ros2',
              },
              {
                label: 'Digital Twin Simulation',
                to: '/docs/module2/digital-twin-gazebo-unity',
              },
              {
                label: 'AI-Robot Integration',
                to: '/docs/module3/ai-robot-brain-nvidia-isaac',
              },
              {
                label: 'Vision-Language-Action Systems',
                to: '/docs/module4/vision-language-action-vla',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Stack Overflow',
                href: 'https://stackoverflow.com/questions/tagged/ros2',
              },
              {
                label: 'Robotics Stack Exchange',
                href: 'https://robotics.stackexchange.com/',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/moinuddin247/hackathon-physical-ai-book',
              },
              {
                label: 'Login',
                position: 'right',
                to: '/auth/login',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Physical AI Course. Built with Docusaurus.`,
      },
      prism: {
        theme: require('prism-react-renderer').themes.github,
        darkTheme: require('prism-react-renderer').themes.dracula,
      },
    }),
});