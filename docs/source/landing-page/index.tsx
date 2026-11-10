// SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0

import SiteLink, { LinkIndicator } from "./_links";
import Layout from "@theme/Layout";
import ThemedImage from "@theme/ThemedImage";
import type { ReactElement } from "react";
import { landing } from "./_homepage";
import inkLogo from "./mlia-logo-stacked-ink.png";
import whiteLogo from "./mlia-logo-stacked-white.png";
import styles from "./index.module.css";

export default function Home(): ReactElement {
  return (
    <Layout title={landing.title} description={landing.lead}>
      <main className={styles.surface}>
        <div className={styles.page}>
          <div className={styles.brand}>
            <ThemedImage
              sources={{ light: inkLogo, dark: whiteLogo }}
              alt="Arm machine learning inference advisor"
              width="3300"
              height="432"
            />
          </div>
          <section className={styles.hero} aria-labelledby="landing-title">
            <div className={styles.heroCopy}>
              <p className={styles.eyebrow}>
                {landing.category} / {landing.title}
              </p>
              <h1 id="landing-title">{landing.headline}</h1>
              <p className={styles.lead}>{landing.lead}</p>
              <div className={styles.actions}>
                <SiteLink
                  className={styles.primaryButton}
                  to={landing.learning}
                >
                  Follow the Learning Path
                </SiteLink>
                <SiteLink
                  className={styles.secondaryButton}
                  to="/documentation"
                >
                  Browse documentation
                </SiteLink>
              </div>
              <SiteLink
                className={styles.repositoryLink}
                to={landing.repository}
              >
                View on GitHub
              </SiteLink>
            </div>
          </section>
          <section
            className={styles.learningPath}
            aria-labelledby="learning-title"
          >
            <div>
              <p className={styles.eyebrow}>Start here · Arm Learning Path</p>
              <h2 id="learning-title">{landing.learningTitle}</h2>
              <p>{landing.learningDescription}</p>
            </div>
            <SiteLink className={styles.primaryButton} to={landing.learning}>
              Follow the step-by-step guide
            </SiteLink>
          </section>
          <section
            className={styles.analysisSection}
            aria-labelledby="focus-title"
          >
            <div className={styles.analysisHeader}>
              <p className={styles.eyebrow}>{landing.category}</p>
              <h2 id="focus-title">Work with {landing.title}.</h2>
            </div>
            <div className={styles.capabilities}>
              {landing.focus.map((item, index) => (
                <SiteLink
                  className={styles.capabilityCard}
                  indicator={false}
                  to={item.to}
                  key={item.to}
                >
                  <span className={styles.number}>0{index + 1}</span>
                  <h2>{item.title}</h2>
                  <p>{item.description}</p>
                  <span className={styles.cardLink}>
                    Read the guide <LinkIndicator to={item.to} />
                  </span>
                </SiteLink>
              ))}
            </div>
          </section>
          <div className={`${styles.workspace} ${styles.pairedPanels}`}>
            <section
              className={styles.startSection}
              aria-labelledby="integration-title"
            >
              <div>
                <p className={styles.eyebrow}>Part of MLIA</p>
                <h2 id="integration-title">Start with the right setup.</h2>
                <p className={styles.sectionLead}>{landing.note}</p>
              </div>
              <div className={styles.textLinks}>
                <SiteLink to={landing.start}>Usage guide</SiteLink>
                <SiteLink to="https://developer.arm.com/dev-docs/mlia/">
                  MLIA core documentation
                </SiteLink>
              </div>
            </section>
            <section
              className={`${styles.pluginSection} ${styles.taskPanel}`}
              aria-labelledby="guides-title"
            >
              <div className={styles.sectionHeader}>
                <p className={styles.eyebrow}>Documentation</p>
                <h2 id="guides-title">Find your next step.</h2>
              </div>
              <div className={styles.pluginGrid}>
                {landing.guides.map((guide) => (
                  <SiteLink
                    className={styles.guideRow}
                    key={guide.to}
                    to={guide.to}
                  >
                    <span>{guide.title}</span>
                  </SiteLink>
                ))}
              </div>
            </section>
          </div>
          <section
            className={styles.contribute}
            aria-labelledby="development-title"
          >
            <div>
              <p className={styles.eyebrow}>Under the hood</p>
              <h2 id="development-title">Explore the implementation.</h2>
              <p>Find integration details and contributor guidance.</p>
            </div>
            <SiteLink className={styles.secondaryButton} to="/development">
              Development
            </SiteLink>
          </section>
        </div>
      </main>
    </Layout>
  );
}
