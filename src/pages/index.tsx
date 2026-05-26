import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

type Part = {
  number: string;
  title: string;
  description: string;
  to: string;
  ready: boolean;
};

const parts: Part[] = [
  {number: '00', title: 'Tổng quan về Mixture of Experts', description: 'Sparse vs dense, scaling laws, lý do MoE bùng nổ 2023-2025, thuật ngữ cốt lõi và bản đồ toàn series 10 model.', to: '/docs/00-tong-quan/01-overview', ready: true},
  {number: '01', title: 'Foundations: router, routing, balancing', description: 'Gate logits, top-k selection, jitter noise, token-choice vs expert-choice, auxiliary loss, z-loss, aux-free bias, expert capacity, shared experts.', to: '/docs/01-foundations/01-overview', ready: true},
  {number: '02', title: 'HuggingFace MoE infrastructure', description: 'integrations/moe.py walkthrough: ExpertsInterface, use_experts_implementation decorator, batched_mm vs grouped_mm dispatch, load_balancing_loss_func helper.', to: '/docs/intro', ready: false},
  {number: '03', title: 'Model walkthroughs (10 model)', description: 'Line-level của Mixtral, Switch Transformers, DeepSeek-V3, Qwen3-MoE, GPT-OSS, OLMoE, JetMoE, Jamba, NLLB-MoE, PhiMoE kèm bảng so sánh.', to: '/docs/intro', ready: false},
  {number: '04', title: 'Cross-cutting: EP, TP, quant, serving', description: 'Expert parallelism với RouterParallel và sentinels, tensor parallel cho MoE, MXFP4/FP8 per-expert quantization, continuous batching, training recipe.', to: '/docs/intro', ready: false},
  {number: '05', title: 'Design comparison và decision guide', description: 'Matrix so sánh router, balancing, expert design ngang 10 model. Decision tree: khi nào chọn MoE vs dense, top-1 vs top-k, shared vs fine-grained.', to: '/docs/intro', ready: false},
];

function HomepageHeader(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className={styles.heroInner}>
        <Heading as="h1" className={styles.heroTitle}>{siteConfig.title}</Heading>
        <p className={styles.heroTagline}>{siteConfig.tagline}</p>
        <div className={styles.heroButtons}>
          <Link className={`button button--primary button--lg ${styles.heroButton}`} to="/docs/intro">Bắt đầu đọc</Link>
          <Link className={`button button--secondary button--lg ${styles.heroButton}`} to="/docs/00-tong-quan/01-overview">Vào Tổng quan MoE</Link>
        </div>
      </div>
    </header>
  );
}

function PartGrid(): ReactNode {
  return (
    <section className={styles.gridSection}>
      <div className="container">
        <Heading as="h2" className={styles.sectionTitle}>Chuỗi bài giảng đi sâu</Heading>
        <p className={styles.sectionSubtitle}>
          Mục tiêu: bạn đọc xong sẽ tự tin mở <code>modeling_mixtral.py</code>, <code>modeling_deepseek_v3.py</code>, hay <code>integrations/moe.py</code> và hiểu từng quyết định thiết kế. Mỗi model đều có walkthrough source code thật, kèm derivation toán học, bảng so sánh, và pitfall.
        </p>
        <div className={styles.grid}>
          {parts.map((p) => (
            <Link key={p.number} to={p.to} className={styles.card}>
              <div className={styles.cardNumber}>PHẦN {p.number}</div>
              <Heading as="h3" className={styles.cardTitle}>{p.title}</Heading>
              <p className={styles.cardDescription}>{p.description}</p>
              <span className={styles.badgeReady}>{p.ready ? 'Đọc phần này' : 'Sắp ra mắt'}</span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

function PhilosophySection(): ReactNode {
  return (
    <section className={styles.philosophy}>
      <div className="container">
        <blockquote className={styles.quote}>
          <p><em>Mixture of Experts biến FFN lớn nhất của Transformer thành một bộ nhiều expert, với router quyết định token nào đi qua expert nào. Ý tưởng đơn giản nhưng decision space rất lớn: top-1 hay top-k, capacity hay dropless, shared expert hay không, aux loss hay bias adjustment, mỗi quyết định phân tách một paradigm.</em></p>
        </blockquote>
        <p className={styles.philosophyText}>
          Chuỗi này đi ngang qua 10 model MoE tiêu biểu trong HuggingFace transformers, từ baseline Mixtral tới state-of-the-art DeepSeek-V3, cùng hạ tầng chia sẻ (<code>integrations/moe.py</code>), expert parallelism, và quantization MXFP4. Mục đích không phải liệt kê API mà để bạn hiểu trải nghiệm thiết kế MoE: đánh đổi, lựa chọn, và cách HF hoá chúng thành code.
        </p>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline as string}>
      <HomepageHeader />
      <main>
        <PartGrid />
        <PhilosophySection />
      </main>
    </Layout>
  );
}
