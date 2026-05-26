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
    {
      type: 'category',
      label: 'Phần 2: HF MoE infrastructure',
      link: {type: 'doc', id: '02-hf-moe-infra/01-overview'},
      collapsed: false,
      items: [
        '02-hf-moe-infra/01-overview',
        '02-hf-moe-infra/02-integrations-moe-py-anatomy',
        '02-hf-moe-infra/03-experts-interface-va-decorator',
        '02-hf-moe-infra/04-batched-mm-vs-grouped-mm',
        '02-hf-moe-infra/05-load-balancing-loss-helper',
      ],
    },
    {
      type: 'category',
      label: 'Phần 3: Model walkthroughs',
      link: {type: 'doc', id: '03-models/01-overview'},
      collapsed: false,
      items: [
        '03-models/01-overview',
        '03-models/02-mixtral',
        '03-models/03-switch-transformers',
        '03-models/04-deepseek-v3',
        '03-models/05-qwen3-moe',
        '03-models/06-gpt-oss',
        '03-models/07-olmoe',
        '03-models/08-jetmoe',
        '03-models/09-jamba',
        '03-models/10-nllb-moe',
        '03-models/11-phimoe',
        '03-models/12-tong-ket-so-sanh',
      ],
    },
    {
      type: 'category',
      label: 'Phần 4: Cross-cutting concerns',
      link: {type: 'doc', id: '04-cross-cutting/01-overview'},
      collapsed: false,
      items: [
        '04-cross-cutting/01-overview',
        '04-cross-cutting/02-expert-parallelism',
        '04-cross-cutting/03-tensor-parallel-with-moe',
        '04-cross-cutting/04-quantization-moe',
        '04-cross-cutting/05-inference-serving-moe',
        '04-cross-cutting/06-training-moe-recipe',
      ],
    },
    {
      type: 'category',
      label: 'Phần 5: Design comparison',
      link: {type: 'doc', id: '05-comparison/01-overview'},
      collapsed: false,
      items: [
        '05-comparison/01-overview',
        '05-comparison/02-routing-comparison-matrix',
        '05-comparison/03-load-balancing-comparison',
        '05-comparison/04-expert-design-comparison',
        '05-comparison/05-khi-nao-dung-moe-vs-dense',
      ],
    },
    {
      type: 'category',
      label: 'Tài nguyên',
      collapsed: true,
      items: [
        'resources/glossary',
        'resources/cheatsheet',
        'resources/references',
      ],
    },
  ],
};

export default sidebars;
