import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/hackathon-physical-ai-book/__docusaurus/debug',
    component: ComponentCreator('/hackathon-physical-ai-book/__docusaurus/debug', 'ac7'),
    exact: true
  },
  {
    path: '/hackathon-physical-ai-book/__docusaurus/debug/config',
    component: ComponentCreator('/hackathon-physical-ai-book/__docusaurus/debug/config', 'b61'),
    exact: true
  },
  {
    path: '/hackathon-physical-ai-book/__docusaurus/debug/content',
    component: ComponentCreator('/hackathon-physical-ai-book/__docusaurus/debug/content', '743'),
    exact: true
  },
  {
    path: '/hackathon-physical-ai-book/__docusaurus/debug/globalData',
    component: ComponentCreator('/hackathon-physical-ai-book/__docusaurus/debug/globalData', '4f7'),
    exact: true
  },
  {
    path: '/hackathon-physical-ai-book/__docusaurus/debug/metadata',
    component: ComponentCreator('/hackathon-physical-ai-book/__docusaurus/debug/metadata', '5ec'),
    exact: true
  },
  {
    path: '/hackathon-physical-ai-book/__docusaurus/debug/registry',
    component: ComponentCreator('/hackathon-physical-ai-book/__docusaurus/debug/registry', 'fb3'),
    exact: true
  },
  {
    path: '/hackathon-physical-ai-book/__docusaurus/debug/routes',
    component: ComponentCreator('/hackathon-physical-ai-book/__docusaurus/debug/routes', 'e7f'),
    exact: true
  },
  {
    path: '/hackathon-physical-ai-book/auth/background',
    component: ComponentCreator('/hackathon-physical-ai-book/auth/background', 'd24'),
    exact: true
  },
  {
    path: '/hackathon-physical-ai-book/auth/login',
    component: ComponentCreator('/hackathon-physical-ai-book/auth/login', 'b27'),
    exact: true
  },
  {
    path: '/hackathon-physical-ai-book/docs',
    component: ComponentCreator('/hackathon-physical-ai-book/docs', 'cdf'),
    exact: true
  },
  {
    path: '/hackathon-physical-ai-book/docs',
    component: ComponentCreator('/hackathon-physical-ai-book/docs', '110'),
    routes: [
      {
        path: '/hackathon-physical-ai-book/docs',
        component: ComponentCreator('/hackathon-physical-ai-book/docs', '56e'),
        routes: [
          {
            path: '/hackathon-physical-ai-book/docs',
            component: ComponentCreator('/hackathon-physical-ai-book/docs', '940'),
            routes: [
              {
                path: '/hackathon-physical-ai-book/docs/module1/robotic-nervous-system-ros2',
                component: ComponentCreator('/hackathon-physical-ai-book/docs/module1/robotic-nervous-system-ros2', 'c92'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/hackathon-physical-ai-book/docs/module2/digital-twin-gazebo-unity',
                component: ComponentCreator('/hackathon-physical-ai-book/docs/module2/digital-twin-gazebo-unity', '6b4'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/hackathon-physical-ai-book/docs/module3/ai-robot-brain-nvidia-isaac',
                component: ComponentCreator('/hackathon-physical-ai-book/docs/module3/ai-robot-brain-nvidia-isaac', 'b43'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/hackathon-physical-ai-book/docs/module4/vision-language-action-vla',
                component: ComponentCreator('/hackathon-physical-ai-book/docs/module4/vision-language-action-vla', '902'),
                exact: true,
                sidebar: "docs"
              },
              {
                path: '/hackathon-physical-ai-book/docs/rag-interface',
                component: ComponentCreator('/hackathon-physical-ai-book/docs/rag-interface', '24b'),
                exact: true,
                sidebar: "docs"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/hackathon-physical-ai-book/',
    component: ComponentCreator('/hackathon-physical-ai-book/', '857'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
