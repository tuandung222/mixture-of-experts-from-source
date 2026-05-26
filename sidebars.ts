import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  lectureSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Giới thiệu chuỗi bài giảng',
    },
    {
      type: 'category',
      label: 'Phần 0: Tổng quan về MoE',
      link: {type: 'doc', id: '00-tong-quan/01-overview'},
      collapsed: false,
      items: [
        '00-tong-quan/01-overview',
        '00-tong-quan/02-vi-sao-mixture-of-expert',
        '00-tong-quan/03-thuat-ngu-cot-loi',
        '00-tong-quan/04-roadmap',
      ],
    },
    {
      type: 'category',
      label: 'Phần 1: Foundations',
      link: {type: 'doc', id: '01-foundations/01-overview'},
      collapsed: false,
      items: [
        '01-foundations/01-overview',
        '01-foundations/02-router-anatomy',
        '01-foundations/03-routing-strategies',
        '01-foundations/04-load-balancing',
        '01-foundations/05-expert-capacity-va-token-dropping',
        '01-foundations/06-shared-experts-va-fine-grained',
      ],
    },
  ],
};

export default sidebars;
