/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docs: [
    {
      type: 'category',
      label: 'Physical AI & Humanoid Robotics Course',
      items: [
        'module1/robotic-nervous-system-ros2',
        'module2/digital-twin-gazebo-unity',
        'module3/ai-robot-brain-nvidia-isaac',
        'module4/vision-language-action-vla',
        'rag-interface',
      ],
      collapsed: false,
    },
  ],
};

module.exports = sidebars;