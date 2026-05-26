import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Mixture of Experts From Source',
  tagline: 'Phân tích 10 model Mixture of Experts trong HuggingFace transformers, từ router đến expert parallelism và quantization',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
    faster: true,
  },

  url: 'https://tuandung222.github.io',
  baseUrl: '/mixture-of-experts-from-source/',
  organizationName: 'tuandung222',
  projectName: 'mixture-of-experts-from-source',
  trailingSlash: false,
  onBrokenLinks: 'warn',

  headTags: [
    {
      tagName: 'meta',
      attributes: {
        name: 'robots',
        content: 'noindex,nofollow,noarchive,nosnippet',
      },
    },
  ],

  i18n: {
    defaultLocale: 'vi',
    locales: ['vi'],
    localeConfigs: {
      vi: {label: 'Tiếng Việt', htmlLang: 'vi-VN'},
    },
  },

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  themes: ['@docusaurus/theme-mermaid'],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
      type: 'text/css',
      integrity:
        'sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+',
      crossorigin: 'anonymous',
    },
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
          editUrl: 'https://github.com/tuandung222/mixture-of-experts-from-source/edit/main/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          showLastUpdateTime: false,
          numberPrefixParser: false,
        },
        blog: false,
        sitemap: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  plugins: [
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        hashed: true,
        language: ['en', 'vi'],
        indexBlog: false,
        docsRouteBasePath: '/docs',
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    mermaid: {
      theme: {light: 'neutral', dark: 'dark'},
    },
    navbar: {
      title: 'MoE From Source',
      logo: {
        alt: 'Mixture of Experts From Source',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'lectureSidebar',
          position: 'left',
          label: 'Bài giảng',
        },
        {
          to: '/docs/resources/glossary',
          label: 'Thuật ngữ',
          position: 'left',
        },
        {
          href: 'https://github.com/tuandung222/mixture-of-experts-from-source',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Trục nội dung',
          items: [
            {label: 'Tổng quan MoE', to: '/docs/00-tong-quan/01-overview'},
            {label: 'Foundations', to: '/docs/01-foundations/01-overview'},
            {label: 'HF MoE infrastructure', to: '/docs/02-hf-moe-infra/01-overview'},
            {label: 'Model walkthroughs', to: '/docs/03-models/01-overview'},
            {label: 'Cross-cutting', to: '/docs/04-cross-cutting/01-overview'},
            {label: 'Design comparison', to: '/docs/05-comparison/01-overview'},
            {label: 'Mathematical modeling', to: '/docs/06-mathematical-modeling/01-overview'},
          ],
        },
        {
          title: 'Tài nguyên',
          items: [
            {label: 'Glossary', to: '/docs/resources/glossary'},
            {label: 'Cheatsheet', to: '/docs/resources/cheatsheet'},
            {label: 'References', to: '/docs/resources/references'},
          ],
        },
      ],
      copyright: `Bản quyền © ${new Date().getFullYear()} Mixture of Experts From Source. Nội dung đang được biên soạn.`,
    },
    prism: {
      theme: prismThemes.oneLight,
      darkTheme: prismThemes.oneDark,
      additionalLanguages: ['bash', 'json', 'typescript', 'yaml', 'python', 'markdown', 'docker'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
